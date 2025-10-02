#!/usr/bin/env python3
"""
LFW pairs.txt를 pairs.csv로 변환하는 스크립트
"""
import os
import pandas as pd

def convert_pairs_txt_to_csv():
    """pairs.txt를 pairs.csv로 변환"""
    
    # LFW pairs.txt 파일 경로들 (여러 가능한 위치 확인)
    possible_paths = [
        "/mnt/c/Users/Admin/Downloads/pairs.txt",
        "/mnt/c/Users/Admin/Downloads/lfw-deepfunneled/pairs.txt", 
        "/mnt/c/Users/Admin/Downloads/lfw/pairs.txt",
        "pairs.txt"
    ]
    
    pairs_txt_path = None
    for path in possible_paths:
        if os.path.exists(path):
            pairs_txt_path = path
            break
    
    if pairs_txt_path is None:
        print("❌ pairs.txt 파일을 찾을 수 없습니다.")
        print("다음 위치 중 하나에 pairs.txt를 배치해주세요:")
        for path in possible_paths:
            print(f"   {path}")
        print("\npairs.txt는 다음에서 다운로드할 수 있습니다:")
        print("   http://vis-www.cs.umass.edu/lfw/pairs.txt")
        return None
    
    print(f"✅ pairs.txt 파일 발견: {pairs_txt_path}")
    
    # pairs.txt 읽기
    try:
        with open(pairs_txt_path, 'r') as f:
            lines = f.readlines()
        
        print(f"📄 총 {len(lines)}줄 읽음")
        
        # 첫 번째 줄은 헤더 (예: "10	300")
        header = lines[0].strip()
        print(f"📋 헤더: {header}")
        
        data_lines = lines[1:]  # 실제 데이터
        print(f"📊 데이터 라인: {len(data_lines)}개")
        
        # CSV 데이터 준비
        csv_data = []
        
        for line_no, line in enumerate(data_lines, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            
            if len(parts) == 3:
                # 같은 사람 (3개 필드: name, img1, img2)
                person, img1_num, img2_num = parts
                csv_data.append([person, img1_num, img2_num, ''])  # 마지막 필드는 빈 문자열
                
            elif len(parts) == 4:
                # 다른 사람 (4개 필드: name1, img1, name2, img2)
                person1, img1_num, person2, img2_num = parts
                csv_data.append([person1, img1_num, person2, img2_num])
                
            else:
                print(f"⚠️ 라인 {line_no}: 예상치 못한 형식 - {line}")
                continue
        
        print(f"✅ {len(csv_data)}개의 유효한 쌍 변환됨")
        
        # CSV로 저장
        csv_path = "/mnt/c/Users/Admin/Downloads/pairs.csv"
        
        # DataFrame 생성 (헤더 없이)
        df = pd.DataFrame(csv_data)
        
        # CSV 저장
        df.to_csv(csv_path, index=False, header=False)
        
        print(f"💾 CSV 파일 저장: {csv_path}")
        
        # 확인
        if os.path.exists(csv_path):
            print(f"✅ 변환 완료!")
            
            # 샘플 확인
            print(f"\n📋 변환된 CSV 샘플 (처음 5줄):")
            sample_df = pd.read_csv(csv_path, header=None, nrows=5)
            print(sample_df.to_string(index=False, header=False))
            
            return csv_path
        else:
            print(f"❌ CSV 파일 생성 실패")
            return None
            
    except Exception as e:
        print(f"❌ 변환 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔄 LFW pairs.txt → pairs.csv 변환기")
    print("=" * 50)
    
    result = convert_pairs_txt_to_csv()
    
    if result:
        print(f"\n🎉 변환 성공! 이제 LFW evaluation을 실행할 수 있습니다.")
    else:
        print(f"\n💡 pairs.txt를 다운로드하고 다시 실행해주세요:")
        print(f"   wget http://vis-www.cs.umass.edu/lfw/pairs.txt")