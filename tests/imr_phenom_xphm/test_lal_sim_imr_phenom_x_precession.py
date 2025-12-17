"""Unit tests for lal_sim_imr_phenom_x_precession.py"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ripplegw.waveforms.imr_phenom_xphm.lal_sim_imr_phenom_x_precession import (
    imr_phenom_x_rotate_y,
    imr_phenom_x_rotate_z,
    imr_phenom_x_vector_diff,
    imr_phenom_x_vector_dot_product,
    imr_phenom_x_vector_scalar,
)


class TestRotateZ:
    """Test suite for imr_phenom_x_rotate_z function."""

    def test_rotate_z_zero_angle(self):
        """Test that zero angle rotation leaves vector unchanged."""
        vx, vy, vz = 1.0, 2.0, 3.0
        angle = 0.0

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)

        assert jnp.allclose(vx_rot, vx, atol=1e-15)
        assert jnp.allclose(vy_rot, vy, atol=1e-15)
        assert jnp.allclose(vz_rot, vz, atol=1e-15)

    def test_rotate_z_90_degrees(self):
        """Test 90-degree rotation around z-axis.

        A vector (x, y, z) rotated 90 degrees counterclockwise around z
        should become (-y, x, z).
        """
        vx, vy, vz = 1.0, 0.0, 5.0
        angle = jnp.pi / 2.0

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)

        # After 90-degree rotation: (1, 0, 5) -> (0, 1, 5)
        assert jnp.allclose(vx_rot, 0.0, atol=1e-7)
        assert jnp.allclose(vy_rot, 1.0, atol=1e-7)
        assert jnp.allclose(vz_rot, vz, atol=1e-7)

    def test_rotate_z_180_degrees(self):
        """Test 180-degree rotation around z-axis.

        A vector (x, y, z) rotated 180 degrees around z should become (-x, -y, z).
        """
        vx, vy, vz = 1.0, 2.0, 3.0
        angle = jnp.pi

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)

        assert jnp.allclose(vx_rot, -vx, atol=1e-14)
        assert jnp.allclose(vy_rot, -vy, atol=1e-14)
        assert jnp.allclose(vz_rot, vz, atol=1e-14)

    def test_rotate_z_360_degrees(self):
        """Test full 360-degree rotation returns to original vector."""
        vx, vy, vz = 1.5, 2.5, 3.5
        angle = 2.0 * jnp.pi

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)

        assert jnp.allclose(vx_rot, vx, atol=1e-14)
        assert jnp.allclose(vy_rot, vy, atol=1e-14)
        assert jnp.allclose(vz_rot, vz, atol=1e-14)

    def test_rotate_z_preserves_magnitude_xy_plane(self):
        """Test that rotation preserves the magnitude of the xy-component."""
        vx, vy, vz = 3.0, 4.0, 5.0
        angle = jnp.pi / 4.0  # 45 degrees

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)

        # Magnitude in xy plane should be preserved
        mag_orig = jnp.sqrt(vx**2 + vy**2)
        mag_rot = jnp.sqrt(vx_rot**2 + vy_rot**2)

        assert jnp.allclose(mag_orig, mag_rot, atol=1e-14)
        # z-component should be unchanged
        assert jnp.allclose(vz_rot, vz, atol=1e-14)

    def test_rotate_z_preserves_magnitude_overall(self):
        """Test that rotation preserves the overall vector magnitude."""
        vx, vy, vz = 1.0, 2.0, 3.0
        angle = 0.7  # arbitrary angle

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)

        mag_orig = jnp.sqrt(vx**2 + vy**2 + vz**2)
        mag_rot = jnp.sqrt(vx_rot**2 + vy_rot**2 + vz_rot**2)

        assert jnp.allclose(mag_orig, mag_rot, atol=1e-14)

    def test_rotate_z_negative_angle(self):
        """Test that negative angle rotates in opposite direction."""
        vx, vy, vz = 1.0, 0.0, 0.0
        angle = jnp.pi / 2.0

        vx_rot_pos, vy_rot_pos, _vz_rot_pos = imr_phenom_x_rotate_z(angle, vx, vy, vz)
        vx_rot_neg, vy_rot_neg, _vz_rot_neg = imr_phenom_x_rotate_z(-angle, vx, vy, vz)

        # Negative rotation should give y_rot with opposite sign
        assert jnp.allclose(vy_rot_pos, -vy_rot_neg, atol=1e-14)
        assert jnp.allclose(vx_rot_pos, vx_rot_neg, atol=1e-14)

    def test_rotate_z_twice_is_additive(self):
        """Test that two sequential rotations equal a single combined rotation."""
        vx, vy, vz = 2.0, 3.0, 4.0
        angle1 = jnp.pi / 6.0  # 30 degrees
        angle2 = jnp.pi / 3.0  # 60 degrees

        # Sequential rotations
        vx_temp, vy_temp, vz_temp = imr_phenom_x_rotate_z(angle1, vx, vy, vz)
        vx_seq, vy_seq, vz_seq = imr_phenom_x_rotate_z(angle2, vx_temp, vy_temp, vz_temp)

        # Combined rotation
        vx_combined, vy_combined, vz_combined = imr_phenom_x_rotate_z(angle1 + angle2, vx, vy, vz)

        assert jnp.allclose(vx_seq, vx_combined, atol=1e-14)
        assert jnp.allclose(vy_seq, vy_combined, atol=1e-14)
        assert jnp.allclose(vz_seq, vz_combined, atol=1e-14)

    def test_rotate_z_zero_vector(self):
        """Test rotation of zero vector."""
        vx, vy, vz = 0.0, 0.0, 0.0
        angle = 1.5

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)

        assert jnp.allclose(vx_rot, 0.0, atol=1e-15)
        assert jnp.allclose(vy_rot, 0.0, atol=1e-15)
        assert jnp.allclose(vz_rot, 0.0, atol=1e-15)

    def test_rotate_z_along_z_axis(self):
        """Test rotation of vector along z-axis.

        A vector along the z-axis should not change under rotation around z.
        """
        vx, vy, vz = 0.0, 0.0, 5.0
        angle = 1.3

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)

        assert jnp.allclose(vx_rot, 0.0, atol=1e-15)
        assert jnp.allclose(vy_rot, 0.0, atol=1e-15)
        assert jnp.allclose(vz_rot, vz, atol=1e-15)

    def test_rotate_z_arbitrary_angle(self):
        """Test rotation with arbitrary angle using rotation matrix properties."""
        vx, vy, vz = 5.0, 7.0, 2.0
        angle = 0.3

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)

        # Manually compute using rotation matrix for z-axis
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        vx_expected = cos_a * vx - sin_a * vy
        vy_expected = sin_a * vx + cos_a * vy
        vz_expected = vz

        assert jnp.allclose(vx_rot, vx_expected, atol=1e-14)
        assert jnp.allclose(vy_rot, vy_expected, atol=1e-14)
        assert jnp.allclose(vz_rot, vz_expected, atol=1e-14)

    def test_rotate_z_large_angle(self):
        """Test rotation with angle larger than 2π."""
        vx, vy, vz = 1.0, 0.0, 0.0
        angle = 2.5 * jnp.pi  # 450 degrees = 90 degrees

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)

        # 450 degrees is equivalent to 90 degrees
        assert jnp.allclose(vx_rot, 0.0, atol=1e-6)
        assert jnp.allclose(vy_rot, 1.0, atol=1e-6)
        assert jnp.allclose(vz_rot, vz, atol=1e-6)

    def test_rotate_z_negative_components(self):
        """Test rotation with negative vector components."""
        vx, vy, vz = -1.0, -2.0, -3.0
        angle = jnp.pi / 4.0

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)

        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        vx_expected = cos_a * vx - sin_a * vy
        vy_expected = sin_a * vx + cos_a * vy
        vz_expected = vz

        assert jnp.allclose(vx_rot, vx_expected, atol=1e-14)
        assert jnp.allclose(vy_rot, vy_expected, atol=1e-14)
        assert jnp.allclose(vz_rot, vz_expected, atol=1e-14)

    def test_rotate_z_jittable(self):
        """Test that imr_phenom_x_rotate_z is JAX jittable."""
        vx, vy, vz = 1.0, 2.0, 3.0
        angle = jnp.pi / 4.0

        # Create a jitted version of the function
        jitted_rotate = jax.jit(imr_phenom_x_rotate_z)

        # Test that it compiles and runs
        vx_rot, vy_rot, vz_rot = jitted_rotate(angle, vx, vy, vz)

        # Verify result is correct
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        vx_expected = cos_a * vx - sin_a * vy
        vy_expected = sin_a * vx + cos_a * vy
        vz_expected = vz

        assert jnp.allclose(vx_rot, vx_expected, atol=1e-14)
        assert jnp.allclose(vy_rot, vy_expected, atol=1e-14)
        assert jnp.allclose(vz_rot, vz_expected, atol=1e-14)

    def test_rotate_z_jittable_with_arrays(self):
        """Test that imr_phenom_x_rotate_z works with JAX arrays when jitted."""
        # Use JAX arrays instead of Python floats
        angles = jnp.array([0.0, jnp.pi / 4.0, jnp.pi / 2.0, jnp.pi])
        vx_vals = jnp.array([1.0, 2.0, 3.0, 4.0])
        vy_vals = jnp.array([0.5, 1.5, 2.5, 3.5])
        vz_vals = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Create a jitted function that processes each rotation
        @jax.jit
        def batch_rotate(angles, vx_vals, vy_vals, vz_vals):
            results = jax.vmap(imr_phenom_x_rotate_z)(angles, vx_vals, vy_vals, vz_vals)
            return results

        vx_rot, vy_rot, vz_rot = batch_rotate(angles, vx_vals, vy_vals, vz_vals)

        # Verify shapes
        assert vx_rot.shape == angles.shape
        assert vy_rot.shape == angles.shape
        assert vz_rot.shape == angles.shape

        # Verify first result (zero angle)
        assert jnp.allclose(vx_rot[0], vx_vals[0], atol=1e-14)
        assert jnp.allclose(vy_rot[0], vy_vals[0], atol=1e-14)
        assert jnp.allclose(vz_rot[0], vz_vals[0], atol=1e-14)

        # Verify magnitude is preserved for all rotations
        mag_orig = jnp.sqrt(vx_vals**2 + vy_vals**2 + vz_vals**2)
        mag_rot = jnp.sqrt(vx_rot**2 + vy_rot**2 + vz_rot**2)
        assert jnp.allclose(mag_orig, mag_rot, atol=1e-14)

    def test_rotate_z_jittable_composition(self):
        """Test that jitted function works correctly with function composition."""
        vx, vy, vz = 2.0, 3.0, 4.0
        angle1 = jnp.pi / 6.0
        angle2 = jnp.pi / 3.0

        # Create a jitted wrapper that does sequential rotations
        @jax.jit
        def double_rotate(angle1, angle2, vx, vy, vz):
            vx1, vy1, vz1 = imr_phenom_x_rotate_z(angle1, vx, vy, vz)
            vx2, vy2, vz2 = imr_phenom_x_rotate_z(angle2, vx1, vy1, vz1)
            return vx2, vy2, vz2

        vx_rot, vy_rot, vz_rot = double_rotate(angle1, angle2, vx, vy, vz)

        # Compare with single rotation by combined angle
        vx_combined, vy_combined, vz_combined = imr_phenom_x_rotate_z(angle1 + angle2, vx, vy, vz)

        assert jnp.allclose(vx_rot, vx_combined, atol=1e-14)
        assert jnp.allclose(vy_rot, vy_combined, atol=1e-14)
        assert jnp.allclose(vz_rot, vz_combined, atol=1e-14)

    def test_rotate_z_jit_gradient(self):
        """Test that gradient computation works with jitted function."""

        # Create a loss function based on the rotation
        def loss_fn(angle, vx, vy, vz):
            vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_z(angle, vx, vy, vz)
            return vx_rot**2 + vy_rot**2 + vz_rot**2

        # Jit the loss function
        jitted_loss = jax.jit(loss_fn)

        # Compute gradient
        angle = jnp.pi / 4.0
        vx, vy, vz = 1.0, 2.0, 3.0

        grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
        grad_angle = grad_fn(angle, vx, vy, vz)

        # Verify gradient is a JAX array
        assert isinstance(grad_angle, jax.Array)

        # Verify loss is computed correctly
        loss_value = jitted_loss(angle, vx, vy, vz)
        expected_loss = vx**2 + vy**2 + vz**2  # Rotation preserves magnitude
        assert jnp.allclose(loss_value, expected_loss, atol=1e-14)


class TestRotateY:
    """Test suite for imr_phenom_x_rotate_y function."""

    def test_rotate_y_zero_angle(self):
        """Test that zero angle rotation leaves vector unchanged."""
        vx, vy, vz = 1.0, 2.0, 3.0
        angle = 0.0

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)

        assert jnp.allclose(vx_rot, vx, atol=1e-15)
        assert jnp.allclose(vy_rot, vy, atol=1e-15)
        assert jnp.allclose(vz_rot, vz, atol=1e-15)

    def test_rotate_y_90_degrees(self):
        """Test 90-degree rotation around y-axis.

        A vector (x, y, z) rotated 90 degrees counterclockwise around y
        should become (z, y, -x).
        """
        vx, vy, vz = 1.0, 2.0, 0.0
        angle = jnp.pi / 2.0

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)

        # After 90-degree rotation: (1, 2, 0) -> (0, 2, -1)
        assert jnp.allclose(vx_rot, 0.0, atol=1e-7)
        assert jnp.allclose(vy_rot, vy, atol=1e-7)
        assert jnp.allclose(vz_rot, -1.0, atol=1e-7)

    def test_rotate_y_180_degrees(self):
        """Test 180-degree rotation around y-axis.

        A vector (x, y, z) rotated 180 degrees around y should become (-x, y, -z).
        """
        vx, vy, vz = 1.0, 2.0, 3.0
        angle = jnp.pi

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)

        assert jnp.allclose(vx_rot, -vx, atol=1e-14)
        assert jnp.allclose(vy_rot, vy, atol=1e-14)
        assert jnp.allclose(vz_rot, -vz, atol=1e-14)

    def test_rotate_y_360_degrees(self):
        """Test full 360-degree rotation returns to original vector."""
        vx, vy, vz = 1.5, 2.5, 3.5
        angle = 2.0 * jnp.pi

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)

        assert jnp.allclose(vx_rot, vx, atol=1e-14)
        assert jnp.allclose(vy_rot, vy, atol=1e-14)
        assert jnp.allclose(vz_rot, vz, atol=1e-14)

    def test_rotate_y_preserves_magnitude_xz_plane(self):
        """Test that rotation preserves the magnitude of the xz-component."""
        vx, vy, vz = 3.0, 4.0, 5.0
        angle = jnp.pi / 4.0  # 45 degrees

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)

        # Magnitude in xz plane should be preserved
        mag_orig = jnp.sqrt(vx**2 + vz**2)
        mag_rot = jnp.sqrt(vx_rot**2 + vz_rot**2)

        assert jnp.allclose(mag_orig, mag_rot, atol=1e-14)
        # y-component should be unchanged
        assert jnp.allclose(vy_rot, vy, atol=1e-14)

    def test_rotate_y_preserves_magnitude_overall(self):
        """Test that rotation preserves the overall vector magnitude."""
        vx, vy, vz = 1.0, 2.0, 3.0
        angle = 0.7  # arbitrary angle

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)

        mag_orig = jnp.sqrt(vx**2 + vy**2 + vz**2)
        mag_rot = jnp.sqrt(vx_rot**2 + vy_rot**2 + vz_rot**2)

        assert jnp.allclose(mag_orig, mag_rot, atol=1e-14)

    def test_rotate_y_negative_angle(self):
        """Test that negative angle rotates in opposite direction."""
        vx, vy, vz = 1.0, 2.0, 0.0
        angle = jnp.pi / 2.0

        vx_rot_pos, _vy_rot_pos, vz_rot_pos = imr_phenom_x_rotate_y(angle, vx, vy, vz)
        vx_rot_neg, _vy_rot_neg, vz_rot_neg = imr_phenom_x_rotate_y(-angle, vx, vy, vz)

        # Negative rotation should give vz_rot with opposite sign
        assert jnp.allclose(vz_rot_pos, -vz_rot_neg, atol=1e-14)
        assert jnp.allclose(vx_rot_pos, vx_rot_neg, atol=1e-14)

    def test_rotate_y_twice_is_additive(self):
        """Test that two sequential rotations equal a single combined rotation."""
        vx, vy, vz = 2.0, 3.0, 4.0
        angle1 = jnp.pi / 6.0  # 30 degrees
        angle2 = jnp.pi / 3.0  # 60 degrees

        # Sequential rotations
        vx_temp, vy_temp, vz_temp = imr_phenom_x_rotate_y(angle1, vx, vy, vz)
        vx_seq, vy_seq, vz_seq = imr_phenom_x_rotate_y(angle2, vx_temp, vy_temp, vz_temp)

        # Combined rotation
        vx_combined, vy_combined, vz_combined = imr_phenom_x_rotate_y(angle1 + angle2, vx, vy, vz)

        assert jnp.allclose(vx_seq, vx_combined, atol=1e-14)
        assert jnp.allclose(vy_seq, vy_combined, atol=1e-14)
        assert jnp.allclose(vz_seq, vz_combined, atol=1e-14)

    def test_rotate_y_zero_vector(self):
        """Test rotation of zero vector."""
        vx, vy, vz = 0.0, 0.0, 0.0
        angle = 1.5

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)

        assert jnp.allclose(vx_rot, 0.0, atol=1e-15)
        assert jnp.allclose(vy_rot, 0.0, atol=1e-15)
        assert jnp.allclose(vz_rot, 0.0, atol=1e-15)

    def test_rotate_y_along_y_axis(self):
        """Test rotation of vector along y-axis.

        A vector along the y-axis should not change under rotation around y.
        """
        vx, vy, vz = 0.0, 5.0, 0.0
        angle = 1.3

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)

        assert jnp.allclose(vx_rot, 0.0, atol=1e-15)
        assert jnp.allclose(vy_rot, vy, atol=1e-15)
        assert jnp.allclose(vz_rot, 0.0, atol=1e-15)

    def test_rotate_y_arbitrary_angle(self):
        """Test rotation with arbitrary angle using rotation matrix properties."""
        vx, vy, vz = 5.0, 7.0, 2.0
        angle = 0.3

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)

        # Manually compute using rotation matrix for y-axis
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        vx_expected = cos_a * vx + sin_a * vz
        vy_expected = vy
        vz_expected = -sin_a * vx + cos_a * vz

        assert jnp.allclose(vx_rot, vx_expected, atol=1e-14)
        assert jnp.allclose(vy_rot, vy_expected, atol=1e-14)
        assert jnp.allclose(vz_rot, vz_expected, atol=1e-14)

    def test_rotate_y_large_angle(self):
        """Test rotation with angle larger than 2π."""
        vx, vy, vz = 1.0, 2.0, 0.0
        angle = 2.5 * jnp.pi  # 450 degrees = 90 degrees

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)

        # 450 degrees is equivalent to 90 degrees
        assert jnp.allclose(vx_rot, 0.0, atol=1e-6)
        assert jnp.allclose(vy_rot, vy, atol=1e-6)
        assert jnp.allclose(vz_rot, -1.0, atol=1e-6)

    def test_rotate_y_negative_components(self):
        """Test rotation with negative vector components."""
        vx, vy, vz = -1.0, -2.0, -3.0
        angle = jnp.pi / 4.0

        vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)

        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        vx_expected = cos_a * vx + sin_a * vz
        vy_expected = vy
        vz_expected = -sin_a * vx + cos_a * vz

        assert jnp.allclose(vx_rot, vx_expected, atol=1e-14)
        assert jnp.allclose(vy_rot, vy_expected, atol=1e-14)
        assert jnp.allclose(vz_rot, vz_expected, atol=1e-14)

    def test_rotate_y_jittable(self):
        """Test that imr_phenom_x_rotate_y is JAX jittable."""
        vx, vy, vz = 1.0, 2.0, 3.0
        angle = jnp.pi / 4.0

        # Create a jitted version of the function
        jitted_rotate = jax.jit(imr_phenom_x_rotate_y)

        # Test that it compiles and runs
        vx_rot, vy_rot, vz_rot = jitted_rotate(angle, vx, vy, vz)

        # Verify result is correct
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        vx_expected = cos_a * vx + sin_a * vz
        vy_expected = vy
        vz_expected = -sin_a * vx + cos_a * vz

        assert jnp.allclose(vx_rot, vx_expected, atol=1e-14)
        assert jnp.allclose(vy_rot, vy_expected, atol=1e-14)
        assert jnp.allclose(vz_rot, vz_expected, atol=1e-14)

    def test_rotate_y_jittable_with_arrays(self):
        """Test that imr_phenom_x_rotate_y works with JAX arrays when jitted."""
        # Use JAX arrays instead of Python floats
        angles = jnp.array([0.0, jnp.pi / 4.0, jnp.pi / 2.0, jnp.pi])
        vx_vals = jnp.array([1.0, 2.0, 3.0, 4.0])
        vy_vals = jnp.array([0.5, 1.5, 2.5, 3.5])
        vz_vals = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Create a jitted function that processes each rotation
        @jax.jit
        def batch_rotate(angles, vx_vals, vy_vals, vz_vals):
            results = jax.vmap(imr_phenom_x_rotate_y)(angles, vx_vals, vy_vals, vz_vals)
            return results

        vx_rot, vy_rot, vz_rot = batch_rotate(angles, vx_vals, vy_vals, vz_vals)

        # Verify shapes
        assert vx_rot.shape == angles.shape
        assert vy_rot.shape == angles.shape
        assert vz_rot.shape == angles.shape

        # Verify first result (zero angle)
        assert jnp.allclose(vx_rot[0], vx_vals[0], atol=1e-14)
        assert jnp.allclose(vy_rot[0], vy_vals[0], atol=1e-14)
        assert jnp.allclose(vz_rot[0], vz_vals[0], atol=1e-14)

        # Verify magnitude is preserved for all rotations
        mag_orig = jnp.sqrt(vx_vals**2 + vy_vals**2 + vz_vals**2)
        mag_rot = jnp.sqrt(vx_rot**2 + vy_rot**2 + vz_rot**2)
        assert jnp.allclose(mag_orig, mag_rot, atol=1e-14)

    def test_rotate_y_jittable_composition(self):
        """Test that jitted function works correctly with function composition."""
        vx, vy, vz = 2.0, 3.0, 4.0
        angle1 = jnp.pi / 6.0
        angle2 = jnp.pi / 3.0

        # Create a jitted wrapper that does sequential rotations
        @jax.jit
        def double_rotate(angle1, angle2, vx, vy, vz):
            vx1, vy1, vz1 = imr_phenom_x_rotate_y(angle1, vx, vy, vz)
            vx2, vy2, vz2 = imr_phenom_x_rotate_y(angle2, vx1, vy1, vz1)
            return vx2, vy2, vz2

        vx_rot, vy_rot, vz_rot = double_rotate(angle1, angle2, vx, vy, vz)

        # Compare with single rotation by combined angle
        vx_combined, vy_combined, vz_combined = imr_phenom_x_rotate_y(angle1 + angle2, vx, vy, vz)

        assert jnp.allclose(vx_rot, vx_combined, atol=1e-14)
        assert jnp.allclose(vy_rot, vy_combined, atol=1e-14)
        assert jnp.allclose(vz_rot, vz_combined, atol=1e-14)

    def test_rotate_y_jit_gradient(self):
        """Test that gradient computation works with jitted function."""

        # Create a loss function based on the rotation
        def loss_fn(angle, vx, vy, vz):
            vx_rot, vy_rot, vz_rot = imr_phenom_x_rotate_y(angle, vx, vy, vz)
            return vx_rot**2 + vy_rot**2 + vz_rot**2

        # Jit the loss function
        jitted_loss = jax.jit(loss_fn)

        # Compute gradient
        angle = jnp.pi / 4.0
        vx, vy, vz = 1.0, 2.0, 3.0

        grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
        grad_angle = grad_fn(angle, vx, vy, vz)

        # Verify gradient is a JAX array
        assert isinstance(grad_angle, jax.Array)

        # Verify loss is computed correctly
        loss_value = jitted_loss(angle, vx, vy, vz)
        expected_loss = vx**2 + vy**2 + vz**2  # Rotation preserves magnitude
        assert jnp.allclose(loss_value, expected_loss, atol=1e-14)


class TestVectorDotProduct:
    """Test suite for imr_phenom_x_vector_dot_product function."""

    def test_vector_dot_product_basic(self):
        """Test basic dot product calculation."""
        v1 = jnp.array([1.0, 2.0, 3.0])
        v2 = jnp.array([4.0, 5.0, 6.0])

        result = imr_phenom_x_vector_dot_product(v1, v2)
        expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0  # 4 + 10 + 18 = 32

        assert jnp.allclose(result, expected, atol=1e-14)

    def test_vector_dot_product_zero_vectors(self):
        """Test dot product with zero vectors."""
        v1 = jnp.array([0.0, 0.0, 0.0])
        v2 = jnp.array([1.0, 2.0, 3.0])

        result = imr_phenom_x_vector_dot_product(v1, v2)
        assert jnp.allclose(result, 0.0, atol=1e-14)

        result = imr_phenom_x_vector_dot_product(v2, v1)
        assert jnp.allclose(result, 0.0, atol=1e-14)

    def test_vector_dot_product_orthogonal(self):
        """Test dot product of orthogonal vectors."""
        # x and y axes are orthogonal
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([0.0, 1.0, 0.0])

        result = imr_phenom_x_vector_dot_product(v1, v2)
        assert jnp.allclose(result, 0.0, atol=1e-14)

        # x and z axes are orthogonal
        v3 = jnp.array([0.0, 0.0, 1.0])
        result = imr_phenom_x_vector_dot_product(v1, v3)
        assert jnp.allclose(result, 0.0, atol=1e-14)

    def test_vector_dot_product_unit_vectors(self):
        """Test dot product of unit vectors."""
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([1.0, 0.0, 0.0])

        result = imr_phenom_x_vector_dot_product(v1, v2)
        assert jnp.allclose(result, 1.0, atol=1e-14)

        # 45-degree angle
        v3 = jnp.array([1.0, 1.0, 0.0])
        v3_norm = v3 / jnp.sqrt(jnp.sum(v3**2))
        result = imr_phenom_x_vector_dot_product(v3_norm, v3_norm)
        assert jnp.allclose(result, 1.0, atol=1e-14)

    def test_vector_dot_product_negative_components(self):
        """Test dot product with negative vector components."""
        v1 = jnp.array([-1.0, 2.0, -3.0])
        v2 = jnp.array([4.0, -5.0, 6.0])

        result = imr_phenom_x_vector_dot_product(v1, v2)
        expected = (-1.0 * 4.0) + (2.0 * -5.0) + (-3.0 * 6.0)  # -4 -10 -18 = -32

        assert jnp.allclose(result, expected, atol=1e-14)

    def test_vector_dot_product_commutative(self):
        """Test that dot product is commutative."""
        v1 = jnp.array([1.0, 2.0, 3.0])
        v2 = jnp.array([4.0, 5.0, 6.0])

        result1 = imr_phenom_x_vector_dot_product(v1, v2)
        result2 = imr_phenom_x_vector_dot_product(v2, v1)

        assert jnp.allclose(result1, result2, atol=1e-14)

    def test_vector_dot_product_jittable(self):
        """Test that imr_phenom_x_vector_dot_product is JAX jittable."""
        v1 = jnp.array([1.0, 2.0, 3.0])
        v2 = jnp.array([4.0, 5.0, 6.0])

        # Create a jitted version of the function
        jitted_dot = jax.jit(imr_phenom_x_vector_dot_product)

        # Test that it compiles and runs
        result = jitted_dot(v1, v2)
        expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0

        assert jnp.allclose(result, expected, atol=1e-14)

    def test_vector_dot_product_jittable_with_arrays(self):
        """Test that imr_phenom_x_vector_dot_product works with JAX arrays when jitted."""
        # Create arrays of vectors
        v1_array = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        v2_array = jnp.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

        # Create a jitted function that processes each dot product
        @jax.jit
        def batch_dot(v1_array, v2_array):
            results = jax.vmap(imr_phenom_x_vector_dot_product)(v1_array, v2_array)
            return results

        results = batch_dot(v1_array, v2_array)

        # Verify results
        expected1 = 1.0 * 7.0 + 2.0 * 8.0 + 3.0 * 9.0  # 7 + 16 + 27 = 50
        expected2 = 4.0 * 10.0 + 5.0 * 11.0 + 6.0 * 12.0  # 40 + 55 + 72 = 167

        assert jnp.allclose(results[0], expected1, atol=1e-14)
        assert jnp.allclose(results[1], expected2, atol=1e-14)

    def test_vector_dot_product_jit_gradient(self):
        """Test that gradient computation works with jitted function."""

        # Create a loss function based on the dot product
        def loss_fn(v1, v2):
            return imr_phenom_x_vector_dot_product(v1, v2)

        # Compute gradient with respect to first vector
        v1 = jnp.array([1.0, 2.0, 3.0])
        v2 = jnp.array([4.0, 5.0, 6.0])

        grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
        grad_v1 = grad_fn(v1, v2)

        # Gradient of dot product w.r.t. v1 should be v2
        assert jnp.allclose(grad_v1, v2, atol=1e-14)

        # Compute gradient with respect to second vector
        grad_fn_v2 = jax.jit(jax.grad(loss_fn, argnums=1))
        grad_v2 = grad_fn_v2(v1, v2)

        # Gradient of dot product w.r.t. v2 should be v1
        assert jnp.allclose(grad_v2, v1, atol=1e-14)


class TestVectorScalar:
    """Test imr_phenom_x_vector_scalar function."""

    def test_vector_scalar_basic(self):
        """Test basic scalar multiplication."""
        v1 = jnp.array([1.0, 2.0, 3.0])
        a = 2.0
        result = imr_phenom_x_vector_scalar(v1, a)
        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(result, expected)

    def test_vector_scalar_zero_scalar(self):
        """Test multiplication by zero."""
        v1 = jnp.array([1.0, 2.0, 3.0])
        a = 0.0
        result = imr_phenom_x_vector_scalar(v1, a)
        expected = jnp.array([0.0, 0.0, 0.0])
        assert jnp.allclose(result, expected)

    def test_vector_scalar_negative_scalar(self):
        """Test multiplication by negative scalar."""
        v1 = jnp.array([1.0, -2.0, 3.0])
        a = -1.5
        result = imr_phenom_x_vector_scalar(v1, a)
        expected = jnp.array([-1.5, 3.0, -4.5])
        assert jnp.allclose(result, expected)

    def test_vector_scalar_jit_compatible(self):
        """Test that function is JIT-compatible."""

        @jax.jit
        def jitted_scalar(v1, a):
            return imr_phenom_x_vector_scalar(v1, a)

        v1 = jnp.array([1.0, 2.0, 3.0])
        a = 3.0
        result = jitted_scalar(v1, a)
        expected = jnp.array([3.0, 6.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_vector_scalar_vmap_compatible(self):
        """Test that function works with vmap."""
        vmapped_scalar = jax.vmap(imr_phenom_x_vector_scalar, in_axes=(0, 0))

        v1_batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        a_batch = jnp.array([2.0, 3.0])
        result = vmapped_scalar(v1_batch, a_batch)
        expected = jnp.array([[2.0, 4.0], [9.0, 12.0]])
        assert jnp.allclose(result, expected)

    def test_vector_scalar_gradient(self):
        """Test that gradients work correctly."""

        def scalar_func(a):
            v1 = jnp.array([1.0, 2.0, 3.0])
            return jnp.sum(imr_phenom_x_vector_scalar(v1, a))

        grad_func = jax.grad(scalar_func)
        gradient = grad_func(2.0)
        expected = 6.0  # sum([1, 2, 3]) = 6
        assert jnp.isclose(gradient, expected)


class TestVectorDiff:
    """Test suite for imr_phenom_x_vector_diff function."""

    def test_vector_diff_basic(self):
        """Test basic vector subtraction."""
        v1 = jnp.array([5.0, 7.0, 2.0])
        v2 = jnp.array([1.0, 3.0, 4.0])

        result = imr_phenom_x_vector_diff(v1, v2)
        expected = jnp.array([4.0, 4.0, -2.0])  # [5-1, 7-3, 2-4]

        assert jnp.allclose(result, expected, atol=1e-14)

    def test_vector_diff_zero_vectors(self):
        """Test subtraction with zero vectors."""
        v1 = jnp.array([1.0, 2.0, 3.0])
        v2 = jnp.array([0.0, 0.0, 0.0])

        result = imr_phenom_x_vector_diff(v1, v2)
        assert jnp.allclose(result, v1, atol=1e-14)

        result = imr_phenom_x_vector_diff(v2, v1)
        expected = jnp.array([-1.0, -2.0, -3.0])
        assert jnp.allclose(result, expected, atol=1e-14)

    def test_vector_diff_identical_vectors(self):
        """Test subtraction of identical vectors."""
        v1 = jnp.array([1.0, 2.0, 3.0])
        v2 = jnp.array([1.0, 2.0, 3.0])

        result = imr_phenom_x_vector_diff(v1, v2)
        expected = jnp.array([0.0, 0.0, 0.0])

        assert jnp.allclose(result, expected, atol=1e-14)

    def test_vector_diff_negative_components(self):
        """Test subtraction with negative vector components."""
        v1 = jnp.array([-1.0, 2.0, -3.0])
        v2 = jnp.array([4.0, -5.0, 6.0])

        result = imr_phenom_x_vector_diff(v1, v2)
        expected = jnp.array([-5.0, 7.0, -9.0])  # [-1-4, 2-(-5), -3-6]

        assert jnp.allclose(result, expected, atol=1e-14)

    def test_vector_diff_commutative_property(self):
        """Test that v1 - v2 = -(v2 - v1)."""
        v1 = jnp.array([1.0, 2.0, 3.0])
        v2 = jnp.array([4.0, 5.0, 6.0])

        result1 = imr_phenom_x_vector_diff(v1, v2)
        result2 = imr_phenom_x_vector_diff(v2, v1)

        # result2 should be the negative of result1
        assert jnp.allclose(result1, -result2, atol=1e-14)

    def test_vector_diff_jittable(self):
        """Test that imr_phenom_x_vector_diff is JAX jittable."""
        v1 = jnp.array([5.0, 7.0, 2.0])
        v2 = jnp.array([1.0, 3.0, 4.0])

        # Create a jitted version of the function
        jitted_diff = jax.jit(imr_phenom_x_vector_diff)

        # Test that it compiles and runs
        result = jitted_diff(v1, v2)
        expected = jnp.array([4.0, 4.0, -2.0])

        assert jnp.allclose(result, expected, atol=1e-14)

    def test_vector_diff_jittable_with_arrays(self):
        """Test that imr_phenom_x_vector_diff works with JAX arrays when jitted."""
        # Create arrays of vectors
        v1_array = jnp.array([[5.0, 7.0, 2.0], [8.0, 9.0, 10.0]])
        v2_array = jnp.array([[1.0, 3.0, 4.0], [2.0, 3.0, 5.0]])

        # Create a jitted function that processes each difference
        @jax.jit
        def batch_diff(v1_array, v2_array):
            results = jax.vmap(imr_phenom_x_vector_diff)(v1_array, v2_array)
            return results

        results = batch_diff(v1_array, v2_array)

        # Verify results
        expected1 = jnp.array([4.0, 4.0, -2.0])  # [5-1, 7-3, 2-4]
        expected2 = jnp.array([6.0, 6.0, 5.0])  # [8-2, 9-3, 10-5]

        assert jnp.allclose(results[0], expected1, atol=1e-14)
        assert jnp.allclose(results[1], expected2, atol=1e-14)

    def test_vector_diff_jit_gradient(self):
        """Test that gradient computation works with jitted function."""

        # Create a loss function based on the vector difference
        def diff_func(v1, v2):
            diff = imr_phenom_x_vector_diff(v1, v2)
            return jnp.sum(diff**2)

        # Compute gradient with respect to first vector
        v1 = jnp.array([1.0, 2.0, 3.0])
        v2 = jnp.array([4.0, 5.0, 6.0])

        grad_fn = jax.jit(jax.grad(diff_func, argnums=0))
        grad_v1 = grad_fn(v1, v2)

        # Gradient of sum((v1-v2)^2) w.r.t. v1 should be 2*(v1-v2)
        expected_grad_v1 = 2.0 * (v1 - v2)
        assert jnp.allclose(grad_v1, expected_grad_v1, atol=1e-14)

        # Compute gradient with respect to second vector
        grad_fn_v2 = jax.jit(jax.grad(diff_func, argnums=1))
        grad_v2 = grad_fn_v2(v1, v2)

        # Gradient of sum((v1-v2)^2) w.r.t. v2 should be -2*(v1-v2)
        expected_grad_v2 = -2.0 * (v1 - v2)
        assert jnp.allclose(grad_v2, expected_grad_v2, atol=1e-14)
