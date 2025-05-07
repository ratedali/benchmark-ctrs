from abc import ABC, abstractmethod

import torch

from benchmark_ctrs.certification.parameters import CertificationParameters


class BaseCertificationMethod(ABC):
    """
    Base class for certification methods.
    """

    def create_instance(
        self,
        params: CertificationParameters,
    ):
        self.params = params

    @abstractmethod
    def certify(self, model, device: torch.device):
        """
        Certify the model.

        Args:
            model: The model to be certified.
            device: The device to run the certification on.

        Returns:
            A boolean indicating whether the model is certified or not.
        """
        ...
