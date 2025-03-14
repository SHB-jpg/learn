# -*- coding: utf-8 -*-
import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
import torch

tokenizer = BertTokenizer.from_pretrained("D:\\bert-base-chinese")  # 加载base模型的对应的切词器
model = BertModel.from_pretrained("D:\\bert-base-chinese")

# ====================
# 1. 四元组解析函数
# ====================
def parse_quadruples(output_str):
    """解析多四元组结构"""
    # 去除结尾标记并分割四元组
    quads = output_str.replace(" [END]", "").split(" [SEP] ")

    parsed = []
    for quad in quads:
        parts = [p.strip() for p in quad.split(" | ")]
        if len(parts) != 4:
            continue  # 跳过格式错误的数据

        parsed.append({
            "target": parts[0] if parts[0] != "NULL" else None,
            "argument": parts[1],
            "targeted_group": parts[2],
            "hateful": 1 if parts[3].lower() == "hate" else 0
        })
    return parsed

# ====================
# 2. 数据加载与转换
# ====================
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data=json.load(f)

    processed = []
    for item in data:
        # 解析原始输出
        quads = parse_quadruples(item['output'])

        # 构建平铺结构
        for idx, quad in enumerate(quads):
            processed.append({
                "id": f"{item['id']}_{idx}",
                "text": item['content'],
                "target": quad['target'],
                "argument": quad['argument'],
                "targeted_group": quad['targeted_group'],
                "hateful": quad['hateful'],
                "original_output": item['output']
            })

    return pd.DataFrame(processed)

# ====================
# 3. 数据验证与清洗
# ====================
def validate_data(df):

    # 检查论点是否存在于原文
    def argument_in_text(row):
        return row['argument'] in row['text'] if row['hateful'] == 1 else True

    mismatch = df[~df.apply(argument_in_text, axis=1)]
    if not mismatch.empty:
        print(f"发现{len(mismatch)}条论点与原文不匹配，示例：")
        print(mismatch.sample(2))

    return df

# ====================
# 4. 特征工程
# ====================
def create_features(df):
    # 目标位置特征
    def get_target_position(row):
        if pd.isnull(row['target']) or row['target'] not in row['text']:
            return -1
        return row['text'].find(row['target']) / len(row['text'])

    df['target_position'] = df.apply(get_target_position, axis=1)

    # 论点位置特征
    def get_argument_position(row):
        if pd.isnull(row['argument']) or row['argument'] not in row['text']:
            return -1
        return row['text'].find(row['argument'])/len(row['text'])

    df['argument_position'] = df.apply(get_argument_position, axis=1)

    # 原文转变为词向量
    def get_textembedding(row):
        text=row['text']
        input_ids = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**input_ids)
        last_hidden_state = outputs.last_hidden_state
        embedding = last_hidden_state.mean(dim=1).squeeze()
        return embedding
    df['text_embedding'] = df.apply(get_textembedding, axis=1)

    # 目标群体多标签编码
    df['targeted_group']=df['targeted_group'].apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer()
    group_matrix = mlb.fit_transform(df['targeted_group'])

    print(group_matrix)
    group_df = pd.DataFrame(group_matrix, columns=mlb.classes_)

    return pd.concat([df, group_df], axis=1)

# ====================
# 主流程
# ====================
if __name__ == "__main__":
    # 加载原始数据
    raw_df = load_data("C:\\Users\\SHB\\OneDrive\\桌面\\Tianchi\\tianchi_data\\train.json")

    # 数据验证
    cleaned_df = validate_data(raw_df)

    # 特征工程
    final_df = create_features(cleaned_df)

    # 保存预处理结果
    final_df.to_csv('preprocessed_data.csv', index=False, encoding='utf_8_sig', )

    # 打印数据概览
    print(f"总样本数: {len(final_df)}")
    print("目标群体分布:")
    print(final_df['targeted_group'].value_counts())
    print("\n仇恨标签分布:")
    print(final_df['hateful'].value_counts())

