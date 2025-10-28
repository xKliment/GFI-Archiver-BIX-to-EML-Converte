# Improved EML Converter (GFI Archiver .bix file format to .eml)

Fast, reliable **.bix to .eml** converter for **GFI Archiver** exports. Built in Python with an optimized engine, GUI, and an on-disk ledger for resumable, large-scale migrations.

> **Why?** GFI Archiver’s built-in export tools are limited and can fail at scale or are just bad. This converter re-implements the MAIS Converter Service.

---

<img width="1266" height="875" alt="image" src="https://github.com/user-attachments/assets/95767f88-2013-4e8f-8dc2-eb84b7cd9721" />



## ✅ Key Features

* **Recursive batch conversion** – scans input trees for `.bix`, writes `.eml` to a mirrored output tree
* **Resumable runs** – tiny **SQLite** ledger skips already-converted files and tracks stats
* **Real-time progress** – dual progress bars (scan + convert), live throughput
* **Safe at scale** – network-friendly throttling options in the GUI

---

## 🚀 Quick Start (GUI)
- Install deps: `pip install -r requirements.txt`
- Run: `python bix_to_eml_gui.py` (the bix_conversion_engine_gui.py needs to be in the same dir as the bix_to_eml_gui.py)
- Select Input (BIX tree) and Output (EML tree), then Start.

Then:

1. Select **Input** directory containing `.bix` files
2. Select **Output** directory for `.eml`
3. Click **Start Conversion** and watch the Overview + Session Statistics update

---

## 💡 How It Works (BIX → EML)

GFI Archiver writes message payloads to .BIX Files with **bit inversion** (akin to reading via an “ENStream” with `invertBits`). The inverted stream:

* often begins with **GZIP magic bytes** `1F 8B`; when present, it’s a normal GZIP member → decompress to EML
* otherwise, the **inverted bytes are already the EML**

**Pseudocode**

```text
read bix_bytes
inverted = byte ^ 0xFF for each byte
if inverted starts with 0x1F 0x8B:
    eml = gzip.decompress(inverted)
else:
    eml = inverted
write eml
```

This matches observed GFI MAIS Conversion behavior and is implemented in `improved_bix_converter.py` and the optimized engine.

---

## 📦 Files & Structure

* `bix_to_eml_gui.py` — cross-platform GUI
* `bix_conversion_engine_gui.py` - engine for the GUI
* `bix_to_eml_cli.py` — simple CLI (single file or batch)
* `requirements.txt` — minimal deps (e.g., `psutil`, optional `sv-ttk`)

---

## 🧰 CLI Usage

Convert a directory recursively:

```bash
python bix_to_eml_cli.py --input "C:\archive\bix" --output "C:\archive\eml"
```

Convert a single file:

```bash
python bix_to_eml_cli.py --input message.bix --output message.eml
```

---

## 📝 Notes

* Keep **Input** and **Output** on stable storage; avoid flaky network shares if possible

---

## ⚖️ Disclaimer

This project re-implements conversion behavior to overcome practical limitations in GFI tooling. You are responsible for ensuring use complies with your licenses, policies, and applicable laws.
