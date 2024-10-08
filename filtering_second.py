import pandas as pd

def main():
    # 定义数据集的类型
    split = ["train", "test", "dev"]

    # 遍历每个数据集并打印数据条数
    for s in split:
        # 构建文件路径
        file_path = f"/mnt/lia/scratch/wenqliu/evaluation/delta_causal/regenerate/{s}.jsonl"
        
        # 读取 JSONL 文件
        df = pd.read_json(file_path, lines=True)
        
        # 打印数据条数
        print(f"{s} 数据数: {len(df)}")

if __name__ == "__main__":
    main()
