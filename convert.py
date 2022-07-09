from io import BytesIO
import numpy as np


def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)

# ----------
# quick test


def test():
    x = np.random.uniform(0, 155, (2, 3)).astype(np.float16)
    b = array_to_bytes(x)
    x1 = bytes_to_array(b)
    assert np.all(x == x1)


if __name__ == '__main__':
    test()
