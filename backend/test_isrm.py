"""ISRM siniflandirma birim testleri."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from main import _isrm_sinifi


def test_verilen_ornekler():
    assert _isrm_sinifi(128.0) == "Çok Yüksek Dayanımlı (R5)"
    assert _isrm_sinifi(52.5)  == "Yüksek Dayanımlı (R4)"
    assert _isrm_sinifi(21.9)  == "Düşük Dayanımlı (R2)"
    assert _isrm_sinifi(140.0) == "Çok Yüksek Dayanımlı (R5)"
    assert _isrm_sinifi(31.8)  == "Orta Dayanımlı (R3)"


def test_sinir_degerleri():
    # Alt sinir dahil, ust sinir haric
    assert _isrm_sinifi(0.5)   == "Son Derece Düşük Dayanımlı (R0)"
    assert _isrm_sinifi(1.0)   == "Çok Düşük Dayanımlı (R1)"   # 1 dahil → R1
    assert _isrm_sinifi(4.99)  == "Çok Düşük Dayanımlı (R1)"
    assert _isrm_sinifi(5.0)   == "Düşük Dayanımlı (R2)"        # 5 dahil → R2
    assert _isrm_sinifi(24.99) == "Düşük Dayanımlı (R2)"
    assert _isrm_sinifi(25.0)  == "Orta Dayanımlı (R3)"         # 25 dahil → R3
    assert _isrm_sinifi(49.99) == "Orta Dayanımlı (R3)"
    assert _isrm_sinifi(50.0)  == "Yüksek Dayanımlı (R4)"       # 50 dahil → R4
    assert _isrm_sinifi(99.99) == "Yüksek Dayanımlı (R4)"
    assert _isrm_sinifi(100.0) == "Çok Yüksek Dayanımlı (R5)"  # 100 dahil → R5
    assert _isrm_sinifi(249.99)== "Çok Yüksek Dayanımlı (R5)"
    assert _isrm_sinifi(250.0) == "Aşırı Yüksek Dayanımlı (R6)" # 250 dahil → R6
    assert _isrm_sinifi(300.0) == "Aşırı Yüksek Dayanımlı (R6)"
