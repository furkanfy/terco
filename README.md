# 📊 Telco Customer Churn Prediction

Bu proje, telekomünikasyon şirketi müşterilerinin hizmeti bırakıp bırakmayacağını (churn) tahmin etmek amacıyla geliştirilmiştir. Makine öğrenmesi algoritmalarıyla veri analizi, dengesiz veri yönetimi ve model optimizasyonu süreçlerini kapsamaktadır.

---

## 🚀 Kullanılan Teknikler

- Veri temizleme ve ön işleme
- Kategorik verilerin sayısallaştırılması (`get_dummies`)
- SMOTE ile dengesiz veri setinin dengelenmesi
- Özellik ölçekleme (`StandardScaler`)
- XGBoost sınıflandırıcı
- GridSearchCV ile hiperparametre optimizasyonu
- ROC AUC eğrisi çizimi
- Modelin `.pkl` olarak dışa aktarımı (`joblib`)

---

## 📂 Veri Kümesi

- **Kaynak:** [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Özellikler:** Müşteri bilgileri, hizmet türleri, fatura bilgileri, abonelik durumu vs.

---

## 🧪 En İyi Model

| Metrik         | Değer |
|----------------|-------|
| Model          | XGBoost (GridSearch ile optimize edildi) |
| F1 Skoru       | 0.90 (yaklaşık) |
| ROC AUC Skoru  | ~0.85+ |
| Kaydedildi     | `xgb_telco_model.pkl` olarak dışa aktarıldı |

---

## 🗂 Proje Yapısı

