# 该脚本用于从多个 Excel 文件中提取影像编号，并在指定的原始数据目录中查找对应的文件夹进行复制。
# 注意结果是不明显炎症案例的**原始数据**

import pandas as pd
import shutil
from pathlib import Path

# --- 配置路径 ---
base_path = Path("~/data/brainMRI/脑膜病变/脑膜病变图像").expanduser()
target_dir = base_path / "不明显炎症"
source_subdirs = ["脑炎", "脑炎次诊", "脑膜炎主诊", "脑膜炎次诊"]

# 扫描当前目录下所有的 xlsx 文件
excel_files = list(Path(".").glob("*.xlsx"))

def collect_from_multiple_excels():
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 汇总所有 Excel 中的影像号
    all_target_ids = set()
    print(f"检测到 {len(excel_files)} 个 Excel 文件: {[f.name for f in excel_files]}")

    for excel_path in excel_files:
        try:
            # 同样读取第三列 (index为2)
            df = pd.read_excel(excel_path, header=None)
            # 过滤掉空值并转为字符串
            raw_ids = df.iloc[:, 2].dropna().astype(str).tolist()
            
            for item in raw_ids:
                # 处理 "/" 分隔
                for p in item.split('/'):
                    clean_id = p.strip()
                    if clean_id:
                        all_target_ids.add(clean_id)
        except Exception as e:
            print(f"读取文件 {excel_path.name} 失败: {e}")

    print(f"去重后共获得 {len(all_target_ids)} 个待搜索影像号。")

    # 2. 一次性遍历原始数据目录
    found_ids = set()
    copy_count = 0

    for sub in source_subdirs:
        current_dir = base_path / sub
        if not current_dir.exists():
            continue
            
        print(f"正在扫描: {sub} ...")
        for patient_folder in current_dir.iterdir():
            if patient_folder.is_dir():
                folder_name = patient_folder.name
                
                # 检查文件夹名是否包含任一 ID
                for tid in all_target_ids:
                    if tid in folder_name:
                        dest_folder = target_dir / folder_name
                        if not dest_folder.exists():
                            shutil.copytree(patient_folder, dest_folder)
                            copy_count += 1
                        found_ids.add(tid)
                        break

    # 3. 结果汇总
    print("\n" + "="*30)
    print(f"所有 Excel 处理完毕！")
    print(f"总计复制文件夹: {copy_count} 个")
    
    missing = all_target_ids - found_ids
    if missing:
        print(f"未找到的影像号共 {len(missing)} 个，请核对。")
        # 可选：将没找到的写到一个 txt 里方便查看
        with open("missing_ids.txt", "w") as f:
            f.write("\n".join(sorted(list(missing))))
        print("未找到的 ID 已保存至 missing_ids.txt")
    else:
        print("所有编号匹配成功！")

if __name__ == "__main__":
    collect_from_multiple_excels()