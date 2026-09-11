from h3vm.relay_packet import aligned_layout


offsets, total = aligned_layout((3, 5, 17), alignment=16)
assert offsets == (0, 16, 32)
assert total == 64
assert all((x % 16) == 0 for x in offsets)

empty_offsets, empty_total = aligned_layout((), alignment=16)
assert empty_offsets == ()
assert empty_total == 0

try:
    aligned_layout((1, -1), alignment=16)
except ValueError:
    pass
else:
    raise AssertionError("negative tensor byte size must fail")

print("H3 VRAM Master relay packet layout contract: OK")
