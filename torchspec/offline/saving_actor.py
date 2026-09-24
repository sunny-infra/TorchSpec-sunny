"""Queue consumer that saves inference outputs instead of training on them."""

from __future__ import annotations

import dataclasses

import ray
import torch

from torchspec.offline.dataset import OfflineDataset
from torchspec.ray.ray_actor import RayActor
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore


@ray.remote(num_cpus=1, num_gpus=0)
class OfflineSavingActor:
    """Consume the normal training queues and persist their tensors."""

    def __init__(self, output_dir, mooncake_config) -> None:
        self.dataset = OfflineDataset(output_dir)
        config = dataclasses.replace(
            mooncake_config,
            local_hostname=RayActor.get_node_ip(),
            global_segment_size=0,
            async_put_pool_size=0,
            enable_gpu_direct=False,
        )
        self.store = EagleMooncakeStore(config)
        self.store.setup(device=torch.device("cpu"))
        self.queues = {}

    def set_queues(self, train_queue, eval_queue) -> None:
        self.queues = {"train": train_queue, "eval": eval_queue}

    def save_from_queue(self, split: str, count: int = 1) -> int:
        """Save exactly ``count`` items and return the number newly written."""
        written = 0
        for _ in range(count):
            sample = self.queues[split].get()
            if not sample.data_id:
                raise ValueError("Offline saving requires TrainSample.data_id")
            dtypes = {
                key: getattr(torch, value.replace("torch.", ""))
                if isinstance(value, str)
                else value
                for key, value in (sample.tensor_dtypes or {}).items()
            }
            metadata = sample.metadata or {}
            manifest = metadata.get("vllm_pp_layer_manifest")
            if manifest is not None:
                tensors = self._load_layer_fragments(sample, manifest, dtypes)
            else:
                tensors = self.store.get(
                    key=sample.mooncake_key,
                    shapes=sample.tensor_shapes,
                    dtypes=dtypes,
                    device=torch.device("cpu"),
                ).to_tensor_dict()
            try:
                written += self.dataset.append(
                    split,
                    data_id=sample.data_id,
                    tensors=tensors,
                    packed_loss_mask=sample.packed_loss_mask,
                    metadata=sample.metadata,
                )
            finally:
                if manifest is None:
                    self.store.remove_eagle3_tensors(
                        sample.mooncake_key,
                        has_last_hidden_states="last_hidden_states" in sample.tensor_shapes,
                        has_target="target" in sample.tensor_shapes,
                    )
        return written

    def _load_layer_fragments(self, sample, manifest, dtypes) -> dict[str, torch.Tensor]:
        """Assemble the per-layer objects the connector publishes.

        TP and PP both write ``{key}_layer{id}_hs`` instead of one packed
        hidden-state object. Each fragment is one captured layer.
        """
        metadata = sample.metadata or {}
        if not metadata.get("vllm_pp_complete", False):
            raise RuntimeError(
                f"Incomplete hidden-state sample {sample.data_id}: "
                f"mooncake_key={sample.mooncake_key}"
            )

        seq_len = sample.tensor_shapes["input_ids"][0]
        hidden_size = sample.tensor_shapes["last_hidden_states"][-1]
        fragment_shapes = {
            "hidden_states": (seq_len, hidden_size),
            "input_ids": (seq_len,),
        }
        fragment_dtypes = {
            "hidden_states": dtypes.get("hidden_states", torch.bfloat16),
            "input_ids": torch.int64,
        }

        hidden_states = []
        last_hidden_states = None
        input_ids = None
        for entry in manifest:
            key = entry["mooncake_key"]
            tensors = (
                self.store.get(
                    key=key,
                    shapes=fragment_shapes,
                    dtypes=fragment_dtypes,
                    device=torch.device("cpu"),
                )
                .to_tensor_dict()
            )
            cloned = {name: tensor.clone() for name, tensor in tensors.items()}
            layer_hidden = cloned["hidden_states"]
            layer_input_ids = cloned["input_ids"]
            if input_ids is None:
                input_ids = layer_input_ids
            elif not torch.equal(input_ids, layer_input_ids):
                raise RuntimeError(f"Hidden-state input_ids mismatch for fragment {key}")

            if entry["role"] == "last_hidden_states":
                last_hidden_states = layer_hidden
            else:
                hidden_states.append(layer_hidden)
            self.store.remove_eagle3_tensors(
                key,
                has_last_hidden_states=False,
                has_target=False,
            )

        if not hidden_states or last_hidden_states is None or input_ids is None:
            raise RuntimeError(f"Invalid hidden-state layer manifest for {sample.mooncake_key}")
        return {
            "hidden_states": torch.cat(hidden_states, dim=-1),
            "last_hidden_states": last_hidden_states,
            "input_ids": input_ids,
        }

    def counts(self) -> dict[str, int]:
        return {split: self.dataset.count(split) for split in ("train", "eval")}

    def close(self) -> None:
        self.store.close()
