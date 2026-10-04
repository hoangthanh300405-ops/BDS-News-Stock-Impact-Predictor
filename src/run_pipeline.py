"""Chạy toàn bộ Tầng 1: P4 (giá) -> P1 (tin) -> P6 (target).   Dùng:  python -m src.run_pipeline"""
import time

from src.data import p1_news, p3_policy, p4_prices, p6_targets

if __name__ == "__main__":
    for name, step in (("P4 giá & lịch phiên", p4_prices.build), ("P1 tin tức & nhãn", p1_news.build),
                       ("P3 luồng tin pháp lý", p3_policy.build), ("P6 target", p6_targets.build)):
        t0 = time.time()
        print(f"===== {name} =====")
        step()
        print(f"      ({time.time() - t0:.1f}s)")
