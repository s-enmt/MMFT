#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON Caption Merger

Script to combine multiple JSON caption files and save captions as a list per image ID

Example usage:
python merge_captions.py -o merged.json -i a.json b.json c.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_json_file(file_path: str) -> Dict[str, Any]:
    """JSONファイルを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"エラー: ファイル '{file_path}' のJSONフォーマットが無効です: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"エラー: ファイル '{file_path}' の読み込み中にエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


def merge_caption_files(input_files: List[str]) -> Dict[str, List[str]]:
    """複数のキャプションファイルを結合する"""
    merged_data = {}
    
    for file_path in input_files:
        print(f"ファイル '{file_path}' を読み込み中...")
        data = load_json_file(file_path)
        
        for image_id, caption in data.items():
            if image_id not in merged_data:
                merged_data[image_id] = []
            
            # キャプションが既にリストの場合はそのまま、文字列の場合はリストに変換
            if isinstance(caption, list):
                merged_data[image_id].extend(caption)
            else:
                merged_data[image_id].append(caption)
    
    return merged_data


def save_merged_data(output_path: str, merged_data: Dict[str, List[str]]) -> None:
    """結合されたデータをJSONファイルに保存する"""
    try:
        # 出力ディレクトリが存在しない場合は作成
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"結合されたデータを '{output_path}' に保存しました。")
    except Exception as e:
        print(f"エラー: ファイル '{output_path}' の保存中にエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='複数のJSONキャプションファイルを結合し、画像IDごとにキャプションをリストで保存します。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  %(prog)s -o merged.json -i a.json b.json
  %(prog)s --output results/merged.json --input file1.json file2.json file3.json
        """
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='出力JSONファイルのパス'
    )
    
    parser.add_argument(
        '-i', '--input',
        nargs='+',
        required=True,
        help='入力JSONファイルのパス（複数指定可能）'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='詳細な出力を表示'
    )
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    for file_path in args.input:
        if not Path(file_path).exists():
            print(f"エラー: 入力ファイル '{file_path}' が存在しません。", file=sys.stderr)
            sys.exit(1)
    
    if args.verbose:
        print(f"入力ファイル: {args.input}")
        print(f"出力ファイル: {args.output}")
    
    # ファイル結合処理
    print(f"{len(args.input)}個のファイルを結合しています...")
    merged_data = merge_caption_files(args.input)
    
    # 結果の統計情報を表示
    total_images = len(merged_data)
    total_captions = sum(len(captions) for captions in merged_data.values())
    print(f"結合完了: {total_images}個の画像、合計{total_captions}個のキャプション")
    
    if args.verbose:
        print("\n画像IDごとのキャプション数:")
        for image_id, captions in sorted(merged_data.items()):
            print(f"  {image_id}: {len(captions)}個のキャプション")
    
    # 結果を保存
    save_merged_data(args.output, merged_data)
    print("処理が完了しました。")


if __name__ == '__main__':
    main()