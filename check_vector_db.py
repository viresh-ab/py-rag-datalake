from vector_store import load_index

def check():
    index, meta = load_index()

    print("Vector DB status")
    print("----------------")
    print(f"Total vectors in index : {index.ntotal}")
    print(f"Total metadata records : {len(meta)}")

    if index.ntotal == 0:
        print("❌ Vector DB is empty")
    elif index.ntotal != len(meta):
        print("⚠️ Vector count and metadata count do not match")
    else:
        print("✅ Vector DB is healthy and ready")

if __name__ == "__main__":
    check()