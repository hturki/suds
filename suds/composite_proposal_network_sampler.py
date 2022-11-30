from typing import Optional, Tuple, List, Callable

import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler


class CompositeProposalNetworkSampler(ProposalNetworkSampler):

    def generate_ray_samples(
            self,
            ray_bundle: Optional[RayBundle],
            static_density_fns: List[Callable],
            dynamic_density_fns: List[Callable],
            static_only: bool,
            dynamic_only: bool,
            filter_fn: Optional[Callable]
    ) -> Tuple[RaySamples, List, List, List, List]:
        weights_list = []
        ray_samples_list = []
        static_weights_list = []
        dynamic_weights_list = []

        self.initial_sampler.training = False
        self.pdf_sampler.training = False

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None

        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)

            if is_prop:
                if not dynamic_only:
                    static_density = static_density_fns[i_level](ray_samples.frustums.get_positions())

                if not static_only:
                    dynamic_density = dynamic_density_fns[i_level](ray_samples.frustums.get_positions())

                if static_only:
                    to_use = static_density
                elif dynamic_only:
                    to_use = dynamic_density
                else:
                    to_use = static_density + dynamic_density

                if filter_fn is not None:
                    to_keep = filter_fn(ray_samples)
                    to_use[to_keep <= 0] = 0

                weights = ray_samples.get_weights(to_use)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)

                if not dynamic_only:
                    static_weights_list.append(ray_samples.get_weights(static_density))

                if not static_only:
                    dynamic_weights_list.append(ray_samples.get_weights(dynamic_density))

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list, static_weights_list, dynamic_weights_list
