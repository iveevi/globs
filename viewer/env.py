import numpy as np
import slangpy as spy


def load_hdr(path):
    image = np.array(spy.Bitmap(str(path)), copy=False).astype(np.float32)
    return np.ascontiguousarray(image[..., :3])


def project_sh(image):
    # Project an equirectangular radiance map onto 9 SH coefficients, scaled by
    # the Lambertian cosine-lobe convolution (A_l / pi) so that
    #   diffuse = albedo * sum_i coeff_i * Y_i(normal).
    h, w = image.shape[:2]
    v = (np.arange(h) + 0.5) / h
    u = (np.arange(w) + 0.5) / w
    theta = v * np.pi
    phi = (u - 0.5) * 2.0 * np.pi
    sin_t = np.sin(theta)[:, None]
    cos_t = np.cos(theta)[:, None]
    sin_p = np.sin(phi)[None, :]
    cos_p = np.cos(phi)[None, :]

    # Direction (y up); matches the shader's dir->uv mapping.
    x = sin_t * sin_p
    y = np.broadcast_to(cos_t, (h, w))
    z = sin_t * cos_p
    domega = (np.pi / h) * (2.0 * np.pi / w) * np.broadcast_to(sin_t, (h, w))

    basis = [
        np.full((h, w), 0.282095, np.float32),
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
    a_over_pi = [1.0, 2 / 3, 2 / 3, 2 / 3, 0.25, 0.25, 0.25, 0.25, 0.25]

    weight = domega[..., None]
    coeffs = np.zeros((9, 4), dtype=np.float32)
    for i in range(9):
        c = (image * (basis[i][..., None] * weight)).reshape(-1, 3).sum(0)
        coeffs[i, :3] = c * a_over_pi[i]
    return coeffs
