import pandas as pd
from sqlalchemy import create_engine
from ydata_profiling import ProfileReport

# Conexão (use create_engine — o pandas recomenda isso em vez da string direta)
engine = create_engine("postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl")

# Query corrigida
query = """
    SELECT
        a.*,
        p.*,
        co.*,
        e.analyte,
        e.classification
    FROM clinical.attendances          a
    JOIN clinical.patients             p  ON p.patient_id     = a.patient_id
    JOIN metrics.clinical_outcomes     co ON co.attendance_id = a.attendance_id
    JOIN metrics.exam_records          e  ON e.attendance_id  = a.attendance_id
    WHERE e.analyte IS NOT NULL
      AND e.classification IS NOT NULL
      AND (co.outcome_at - a.attended_at) >= 0
"""

df = pd.read_sql(query, engine)

ProfileReport(df, title="Relatório MOSAIC").to_file("relatorio.html")
