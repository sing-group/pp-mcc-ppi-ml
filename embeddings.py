from abc import ABC, abstractmethod
import numpy as np

class AbstractMergeEmbeddings(ABC):

    @abstractmethod
    def should_duplicate_labels(self):
        """Determine if labels should be duplicated."""
        pass

    @abstractmethod
    def unpack_embeddings(self, df_embeddings):
        """Unpack and process embeddings from the provided DataFrame."""
        pass

    @abstractmethod
    def unpack_embeddings_test(self, df_embeddings):
        """Unpack and process embeddings for testing purposes."""
        pass

    @abstractmethod
    def __str__(self):
        """Provide a meaningful string representation of the instance."""
        pass

    def __repr__(self):
        return self.__str__()

def create_consecutive_groups(n_samples, n_groups):
    group_size = n_samples // n_groups
    remainder = n_samples % n_groups
    groups = [i for i in range(n_groups) for _ in range(group_size)]
    if remainder:
        groups.extend([n_groups - 1] * remainder)
    return groups

class AbstractNumpyMergeEmbeddings(AbstractMergeEmbeddings):

    @abstractmethod
    def merge_embeddings(self, embeddings_1, embeddings_2):
        pass

    def unpack_embeddings(self, df_embeddings):
        rows = [self.merge_embeddings(x1.flatten(), x2.flatten()) for x1, x2 in zip(df_embeddings['emb_prot1'], df_embeddings['emb_prot2'])]
        n_samples = len(rows)
        groups = create_consecutive_groups(len(rows), n_samples)

        return np.array(rows), groups

    def unpack_embeddings_test(self, df_embeddings):
        return self.unpack_embeddings(df_embeddings)

    def should_duplicate_labels(self):
        return False


class AddEmbeddings(AbstractNumpyMergeEmbeddings):

    def __str__(self):
        return f'add'

    def merge_embeddings(self, embeddings_1, embeddings_2):
        return np.add(embeddings_1, embeddings_2)


class MultiplyEmbeddings(AbstractNumpyMergeEmbeddings):

    def __str__(self):
        return f'multiply'
    
    def merge_embeddings(self, embeddings_1, embeddings_2):
        return np.multiply(embeddings_1, embeddings_2)

class DiffEmbeddings(AbstractNumpyMergeEmbeddings):

    def __str__(self):
        return f'diff'
    
    def merge_embeddings(self, embeddings_1, embeddings_2):
        return np.abs(np.subtract(embeddings_1, embeddings_2))

class ConcatEmbeddings(AbstractNumpyMergeEmbeddings):

    def __str__(self):
        return f'concat_'
    
    def merge_embeddings(self, embeddings_1, embeddings_2):
        return np.concatenate((embeddings_1, embeddings_2))

    def unpack_embeddings(self, df_embeddings):
        rows, groups = AbstractNumpyMergeEmbeddings.unpack_embeddings(self, df_embeddings)
        return rows, groups

    def unpack_embeddings_test(self, df_embeddings):
        return AbstractNumpyMergeEmbeddings.unpack_embeddings(self, df_embeddings)
