import gzip
import os
from typing import Optional


def is_gzip(path: str) -> bool:
    """Check if file starts with gzip magic bytes."""
    try:
        with open(path, 'rb') as f:
            b0 = f.read(2)
        return len(b0) == 2 and b0[0] == 0x1F and b0[1] == 0x8B
    except OSError:
        return False


def invert_bits(data: bytes) -> bytes:
    """Invert all bits in the data (equivalent to ENStream with invertBits: true)."""
    return bytes(~b & 0xFF for b in data)


def bix_to_eml(src_path: str, dst_path: Optional[str] = None) -> str:
    """Convert a single .bix file to .eml following MAIS.BixPlugin logic.
    
    Process:
    1. Read BIX file and invert all bits
    2. Check if result is gzip compressed
    3. If compressed, decompress to get EML
    4. If not compressed, the inverted data is the EML
    
    Returns the output .eml path.
    """
    if not dst_path:
        base, _ = os.path.splitext(src_path)
        dst_path = base + ".eml"
    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)

    # Step 1: Read BIX and invert bits (equivalent to ENStream with invertBits: true)
    with open(src_path, 'rb') as f:
        bix_data = f.read()
    
    inverted_data = invert_bits(bix_data)
    
    # Step 2: Check if inverted data is gzip compressed
    if len(inverted_data) >= 2 and inverted_data[0] == 0x1F and inverted_data[1] == 0x8B:
        # Step 3: Decompress to get EML
        try:
            decompressed_data = gzip.decompress(inverted_data)
            with open(dst_path, 'wb') as out:
                out.write(decompressed_data)
        except gzip.BadGzipFile:
            # If decompression fails, write the inverted data as-is
            with open(dst_path, 'wb') as out:
                out.write(inverted_data)
    else:
        # Step 4: Not compressed, inverted data is the EML
        with open(dst_path, 'wb') as out:
            out.write(inverted_data)
    
    return dst_path


def convert_tree(root: str, out_root: Optional[str] = None) -> int:
    """Convert all .bix files under root to .eml under out_root (mirrored).
    Returns the number of converted files.
    """
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith('.bix'):
                continue
            src = os.path.join(dirpath, fn)
            if out_root:
                rel = os.path.relpath(dirpath, root)
                dst_dir = os.path.join(out_root, rel)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, os.path.splitext(fn)[0] + '.eml')
            else:
                dst = os.path.splitext(src)[0] + '.eml'
            bix_to_eml(src, dst)
            count += 1
    return count


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bix_to_eml_cli.py <bix-file-or-directory> [out-root]")
        raise SystemExit(2)
    root = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    
    if os.path.isfile(root):
        # Convert single file
        if not root.lower().endswith('.bix'):
            print("Error: File must have .bix extension")
            raise SystemExit(1)
        bix_to_eml(root, out)
        print(f"Converted {root} to {out or os.path.splitext(root)[0] + '.eml'}")
    elif os.path.isdir(root):
        # Convert directory
        n = convert_tree(root, out)
        print(f"Converted {n} .bix files")
    else:
        print(f"Error: {root} is not a valid file or directory")
        raise SystemExit(1)