import truststore
truststore.inject_into_ssl()

import re
from collections import Counter

from youtube_transcript_api import YouTubeTranscriptApi


# ============================================================
# CONFIGURACIÓN
# ============================================================

video_id = "EJk2Xk9UYrg"


# ============================================================
# OBTENER TRANSCRIPCIÓN
# ============================================================

try:
    api = YouTubeTranscriptApi()

    transcript = api.fetch(
        video_id,
        languages=["es", "en"]
    )

    # Convertir los fragmentos en un solo texto
    transcript_text = " ".join(
        snippet.text for snippet in transcript
    )


    # ========================================================
    # ANÁLISIS
    # ========================================================

    print("=" * 60)
    print("TRANSCRIPCIÓN DE YOUTUBE")
    print("=" * 60)

    print(f"Video ID: {video_id}")
    print(f"Longitud: {len(transcript_text):,} caracteres")
    print(f"Palabras aproximadas: {len(transcript_text.split()):,}")


    # ========================================================
    # PALABRAS MÁS FRECUENTES
    # ========================================================

    words = re.findall(
        r"\b[a-záéíóúñ]{4,}\b",
        transcript_text.lower()
    )

    common_words = Counter(words).most_common(10)

    print("\nPalabras más frecuentes:")

    for word, count in common_words:
        print(f"  {word}: {count} veces")


    # ========================================================
    # PRIMEROS 500 CARACTERES
    # ========================================================

    print("\nPrimeros 500 caracteres:")
    print(transcript_text[:500] + "...")


except Exception as e:

    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print(f"{type(e).__name__}: {e}")