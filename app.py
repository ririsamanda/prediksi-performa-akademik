import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==========================================
# KONFIGURASI APLIKASI & TEMA KUSTOM
# ==========================================
st.set_page_config(
    page_title="Prediksi Performa Akademik",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi CSS untuk mempercantik UI
st.markdown("""
<style>
    /* Styling untuk tombol submit */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    /* Styling untuk kartu metrik */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 10% 5% 10%;
        border-radius: 8px;
    }
    /* Header styling */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "model_performa_akademik.joblib"

# ==========================================
# MEMUAT MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(
        "📁 **File model tidak ditemukan.**\n\n"
        "Letakkan `model_performa_akademik.joblib` "
        "di folder yang sama dengan `app.py`."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Model gagal dimuat: {e}")
    st.stop()

# Mengambil pipeline preprocessing dari model
preprocessor = model.named_steps["preprocessor"]

# Mengambil nama fitur dari pipeline yang sudah dilatih
numeric_features = list(preprocessor.transformers_[0][2])
categorical_features = list(preprocessor.transformers_[1][2])

feature_columns = numeric_features + categorical_features

# Mengambil kategori asli dari OneHotEncoder
categorical_pipeline = preprocessor.named_transformers_["cat"]
encoder = categorical_pipeline.named_steps["encoder"]

category_options = {
    col: [str(value) for value in categories]
    for col, categories in zip(categorical_features, encoder.categories_)
}

# ==========================================
# FUNGSI INPUT NUMERIK
# ==========================================
def numeric_input(col):
    """
    Menampilkan input numerik dengan batas umum.
    Batas ini hanya untuk membantu pengisian formulir.
    """
    bounds = {
        "age": (15.0, 100.0, 20.0),
        "previous_exam_score": (0.0, 100.0, 70.0),
        "previous_gpa": (0.0, 4.0, 3.0),
        "attendance_percentage": (0.0, 100.0, 85.0),
        "assignment_completion_rate": (0.0, 100.0, 80.0),
        "study_hours_per_day": (0.0, 24.0, 3.0),
        "self_study_hours": (0.0, 24.0, 2.0),
        "online_learning_hours": (0.0, 24.0, 1.0),
        "practice_tests_completed": (0.0, 1000.0, 3.0),
        "sleep_hours": (0.0, 24.0, 7.0),
        "daily_screen_time": (0.0, 24.0, 5.0),
        "physical_activity_hours": (0.0, 24.0, 1.0),
        "online_course_hours": (0.0, 24.0, 1.0),
        "time_management_score": (0.0, 100.0, 70.0),
    }

    minimum, maximum, default = bounds.get(col, (0.0, 100.0, 0.0))
    value = min(max(default, minimum), maximum)

    return st.number_input(
        col.replace("_", " ").title(),
        min_value=minimum,
        max_value=maximum,
        value=float(value),
        step=1.0,
        key=f"input_{col}"
    )

# ==========================================
# SIDEBAR DAN NAVIGASI
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=80)
    st.title("AkademikAI")
    st.caption("Sistem Cerdas Klasifikasi Performa Akademik")
    st.divider()
    
    menu = st.radio(
        "📌 Navigasi Utama",
        ["Dashboard", "Prediksi Akademik", "Evaluasi Model"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.caption("© 2026 Proyek Machine Learning")

# ==========================================
# HALAMAN DASHBOARD
# ==========================================
if menu == "Dashboard":
    st.title("🎓 Prediksi Performa Akademik Mahasiswa")
    st.markdown(
        "<p style='font-size: 1.1rem; color: #666;'>"
        "Platform analitik untuk memprediksi tingkat performa akademik "
        "berdasarkan karakteristik mahasiswa dan kebiasaan belajar secara komprehensif.</p>", 
        unsafe_allow_html=True
    )
    
    st.info("💡 **Informasi Model:** Menggunakan algoritma **Support Vector Machine (SVM)** yang dilatih dengan dataset penelitian akademik.", icon="ℹ️")

    st.markdown("### 📊 Ringkasan Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Algoritma Utama", value="SVM Model")
    with col2:
        st.metric(label="Total Fitur Analisis", value=f"{len(feature_columns)} Fitur")
    with col3:
        st.metric(label="Kategori Klasifikasi", value="3 Kelas")

    st.divider()
    st.markdown("### 🎯 Kategori Performa")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("### 🔴 Low")
            st.write("Menunjukkan kategori performa akademik **rendah**. Membutuhkan perhatian dan perbaikan pola belajar.")
    with c2:
        with st.container(border=True):
            st.markdown("### 🟡 Medium")
            st.write("Menunjukkan kategori performa akademik **menengah**. Berada pada batas rata-rata evaluasi.")
    with c3:
        with st.container(border=True):
            st.markdown("### 🟢 High")
            st.write("Menunjukkan kategori performa akademik **tinggi**. Hasil yang sangat optimal dan memuaskan.")

    st.warning(
        "**Catatan:** Dataset yang digunakan bersifat semi-sintetis. "
        "Hasil prediksi merupakan keluaran probabilitas model dan bukan "
        "penilaian mutlak atas kecerdasan atau masa depan seseorang.",
        icon="⚠️"
    )

# ==========================================
# HALAMAN PREDIKSI
# ==========================================
elif menu == "Prediksi Akademik":
    st.title("🔎 Prediksi Performa Akademik")
    st.markdown("Lengkapi formulir di bawah ini dengan profil dan kebiasaan belajar mahasiswa untuk melihat hasil klasifikasi.")

    with st.form("prediction_form", border=True):
        st.subheader("📝 Data Mahasiswa & Kebiasaan Belajar")
        st.write("") # Spacing

        input_values = {}

        # Input numerik (Dibuat menjadi 3 kolom agar lebih rapi)
        if numeric_features:
            st.markdown("**🔢 Informasi Numerik**")
            cols = st.columns(3)
            for i, col in enumerate(numeric_features):
                with cols[i % 3]:
                    input_values[col] = numeric_input(col)
            
            st.write("") # Spacing

        # Input kategorikal
        if categorical_features:
            st.markdown("**🔠 Informasi Kategorikal**")
            cols = st.columns(3)
            for i, col in enumerate(categorical_features):
                options = category_options[col]
                with cols[i % 3]:
                    selected = st.selectbox(
                        col.replace("_", " ").title(),
                        options=options,
                        key=f"select_{col}"
                    )
                    input_values[col] = selected

        st.write("")
        submitted = st.form_submit_button(
            "🚀 Mulai Prediksi Performa",
            type="primary",
            use_container_width=True
        )

    if submitted:
        # Animasi loading agar terkesan sistem sedang berpikir
        with st.spinner("Menganalisis data pola belajar..."):
            input_df = pd.DataFrame([input_values], columns=feature_columns)

            for col in categorical_features:
                input_df[col] = input_df[col].astype(object)

            try:
                prediction = model.predict(input_df)[0]
                
                st.subheader("🎯 Hasil Prediksi Model")
                
                # Kartu hasil prediksi yang dinamis
                if prediction == "High":
                    st.balloons()
                    st.success(
                        "### 🟢 Hasil Klasifikasi: HIGH\n\n"
                        "Luar biasa! Model mengklasifikasikan data ini pada **Kategori Performa Tinggi**. "
                        "Pertahankan ritme dan kebiasaan belajarnya.",
                        icon="🌟"
                    )
                elif prediction == "Medium":
                    st.info(
                        "### 🟡 Hasil Klasifikasi: MEDIUM\n\n"
                        "Model mengklasifikasikan data ini pada **Kategori Performa Menengah**. "
                        "Ada potensi besar untuk ditingkatkan menjadi lebih baik.",
                        icon="📈"
                    )
                elif prediction == "Low":
                    st.error(
                        "### 🔴 Hasil Klasifikasi: LOW\n\n"
                        "Model mengklasifikasikan data ini pada **Kategori Performa Rendah**. "
                        "Disarankan untuk mengevaluasi kembali alokasi waktu dan efektivitas belajarnya.",
                        icon="🚩"
                    )
                else:
                    st.info(f"**Hasil klasifikasi model:** {prediction}")

                with st.expander("Tampilkan Ringkasan Data Input", expanded=False):
                    st.dataframe(input_df, use_container_width=True)

                st.caption(
                    "Disclaimer: Hasil merupakan prediksi probabilistik berdasarkan bobot algoritma SVM, "
                    "bukan jaminan nilai ujian sebenarnya."
                )

            except Exception as e:
                st.error(f"Prediksi gagal diproses: {e}")

# ==========================================
# HALAMAN EVALUASI MODEL
# ==========================================
elif menu == "Evaluasi Model":
    st.title("📊 Evaluasi Model Machine Learning")
    st.markdown("Perbandingan metrik performa antara **Support Vector Machine (SVM)** dan **Gaussian Naive Bayes** pada pengujian *test set*.")

    # Data Hasil
    results = pd.DataFrame({
        "Metrik": ["Accuracy", "Balanced Accuracy", "Macro F1-Score"],
        "SVM": [0.6142, 0.6053, 0.5826],
        "Naive Bayes": [0.6037, 0.5550, 0.5616]
    })

    # Tampilan Tabel yang Lebih Elegan
    st.dataframe(
        results.style.format({
            "SVM": "{:.2%}",
            "Naive Bayes": "{:.2%}"
        }).background_gradient(cmap="Blues", subset=["SVM"])
          .background_gradient(cmap="Oranges", subset=["Naive Bayes"]),
        use_container_width=True,
        hide_index=True
    )

    st.write("")
    st.subheader("📈 Perbandingan Metrik Visual")
    
    with st.container(border=True):
        chart_data = results.set_index("Metrik")
        st.bar_chart(chart_data, height=300)

    st.divider()
    st.subheader("🧩 Confusion Matrix")
    
    # Fungsi kustom untuk menggambar heatmap yang profesional
    def plot_confusion_matrix(cm, title, cmap):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap=cmap, aspect='auto')
        
        # Sembunyikan garis pinggir
        ax.spines[:].set_visible(False)
        
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_yticklabels(["Low", "Medium", "High"])
        ax.set_xlabel("Prediksi Model", fontweight='bold', labelpad=10)
        ax.set_ylabel("Kelas Aktual", fontweight='bold', labelpad=10)
        ax.set_title(title, pad=15, fontweight='bold')

        # Threshold untuk mengubah warna teks agar kontras
        threshold = cm.max() / 2.
        for i in range(3):
            for j in range(3):
                color = "white" if cm[i, j] > threshold else "black"
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color=color, fontweight='bold')
        
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return fig

    # Data Confusion Matrix
    cm_svm = np.array([[7102, 1697, 225],
                       [3191, 3650, 1559],
                       [191, 854, 1531]])

    cm_nb = np.array([[6484, 2383, 157],
                      [2917, 4546, 937],
                      [219, 1313, 1044]])

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            fig_svm = plot_confusion_matrix(cm_svm, "Confusion Matrix - SVM", "Blues")
            st.pyplot(fig_svm)
            plt.close(fig_svm)

    with col2:
        with st.container(border=True):
            fig_nb = plot_confusion_matrix(cm_nb, "Confusion Matrix - Naive Bayes", "Oranges")
            st.pyplot(fig_nb)
            plt.close(fig_nb)

    st.success(
        "**Kesimpulan:** Secara keseluruhan, model **SVM** mencatat performa yang sedikit lebih tinggi "
        "pada metrik Accuracy, Balanced Accuracy, maupun Macro F1-Score dibandingkan dengan Naive Bayes."
    )

    st.caption(
        "Catatan: Metrik evaluasi di atas merupakan catatan statis hasil dari proses *training* dan *testing* "
        "eksperimen data sebelumnya, bukan proses pengujian ulang."
    )