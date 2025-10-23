# DL_Laborario03
Resulción del Lab3 del curso de Deep Learning

Rol
Eres un “Colab Builder” que entrega un notebook funcional, claro y reproducible para el Laboratorio 3 – Aplicación con ASR, LLM y Voice Cloning. Tu salida es un notebook de Google Colab con celdas numeradas, comentarios mínimos pero útiles, y resultados claramente impresos (textos y tiempos).

Objetivo
Construir un pipeline que:

transcriba una pregunta hablada (ASR),

genere una respuesta breve (1–3 oraciones) con un LLM local en Colab, y

sintetice esa respuesta con TTS + Voice Cloning usando la voz de un integrante.
Imprimir tiempos por etapa y total. Entregar función de integración final.

Restricciones clave

Herramientas gratuitas y ejecutables en Google Colab.

LLM ejecutado localmente (transformers en Colab, sin API externa).

Mostrar comparativa de calidad con y sin clonación de voz.

Stack sugerido (puedes mejorar si hay GPU/VRAM):

ASR: OpenAI Whisper (small o turbo según VRAM).

LLM: FLAN-T5-Base (rápido, estable) o Qwen2.5-3B-Instruct si hay GPU y tiempo.

TTS: Coqui TTS xTTS v2 con speaker_wav (voice cloning).
Referencias del enunciado: Whisper, FLAN-T5, Qwen2.5-3B, Coqui TTS.

Estructura del Notebook (celdas numeradas)

Celda 0 — Setup y versiones (pinned)

pip install de: openai-whisper, transformers, accelerate, sentencepiece, coqui-tts, jiwer, soundfile, librosa.

Comprobar CUDA, VRAM, versiones.

Seed para reproducibilidad.

Celda 1 — Utilidades

Cronómetro perf_counter() para tiempos por etapa y total.

Helpers: print_stage_time, ensure_sr_16k_mono(input_wav), save_wav.

Celda 2 — Entrada de audio (robusto en Colab)

Opción A (recomendada): Subir archivo (pregunta) y subir speaker_wav (voz de referencia 10–15 s).

Opción B: Grabador en Colab con widget JS idempotente (ver “Patch Grabador” abajo).

Validar duración de speaker_wav (alerta si <10 s).

Normalizar ambos a 16 kHz mono.

Celda 3 — ASR (Whisper)

Elegir modelo según VRAM (small, base, turbo).

language="es" si las preguntas son en español; si detectas inglés, setear "en".

Guardar: text_transcript, asr_time, num_chars_in.

Celda 4 — LLM local

FLAN-T5-Base con pipeline("text2text-generation").

Prompt mínimo: “Responde en 1–3 oraciones de forma directa y amable. Pregunta: {texto}”.

Guardar: llm_text, llm_time, num_chars_out.

Celda 5 — TTS (xTTS v2 con voice cloning)

Generar dos audios:

Clonado con speaker_wav=<ruta> y language="es".

Base sin clonación (mismo texto) para la comparación requerida.

Guardar: tts_clone_time, tts_base_time.

Celda 6 — Integración end-to-end

Función asr_llm_tts_pipeline(input_wav, speaker_wav, lang="es") que:

cronometre ASR → LLM → TTS(2x),

retorne diccionario con textos, rutas de audios, tiempos por etapa y total.

Imprimir todo de forma limpia (texto transcrito, respuesta del LLM, tiempos).

Celda 7 — Comparativa y demo

Tabla de métricas (asr/llm/tts/total).

Widgets Audio para oír resultado clonado y base.

Observaciones cortas de la calidad percibida (cumple la presentación).

Celda 8 — Empaquetado y checklist de entrega

Confirmar: módulos ASR/LLM/TTS, función integrada, resultados impresos.

Nota final con cómo correr en blanco para el profe (1 bloque de texto).

Reglas de diseño del código

Todo local en Colab (sin claves ni APIs).

Comentarios breves y medibles: imprimir longitud de entrada/salida, tiempos y paths de audios.

Manejo de errores:

Si Whisper falla, sugerir small en vez de turbo.

Si xTTS reporta incompatibilidad, reintentar con language="es" y speaker_wav >10 s.

Reproducible: fijar seeds y versiones, limpiar GPU caché entre corridas opcionalmente.

Patch Grabador (corrige tu error de “Maximum call stack size exceeded”)

Sustituye tu celda de grabación por este recorder idempotente (antes importa from google.colab import output y from IPython.display import Javascript, Audio, display):

from IPython.display import Javascript, Audio, display
from google.colab import output
import base64, subprocess, uuid, os

def record_colab(out_wav="/content/input.wav", sr=16000, autoplay=False):
    js = Javascript(r"""
    async function recorderUIOnce(){
      const EXISTING = document.getElementById('recorder-box');
      if (EXISTING) { EXISTING.remove(); }
      const box = document.createElement('div');
      box.id = 'recorder-box';
      box.style.cssText = 'padding:12px;margin:8px 0;border:1px solid #ddd;border-radius:10px;display:inline-flex;gap:8px;align-items:center;font-family:sans-serif';

      const dot = document.createElement('span');
      dot.style.cssText = 'width:10px;height:10px;border-radius:50%;background:#bbb';

      const startBtn = document.createElement('button');
      startBtn.textContent = 'Grabar';
      startBtn.style.cssText = 'padding:6px 10px';

      const stopBtn = document.createElement('button');
      stopBtn.textContent = 'Parar';
      stopBtn.style.cssText = 'padding:6px 10px';
      stopBtn.disabled = true;

      const msg = document.createElement('span');
      msg.textContent = 'Listo para grabar';
      msg.style.minWidth = '180px';

      box.append(dot, startBtn, stopBtn, msg);
      document.body.appendChild(box);

      let stream, rec, chunks = [];
      function setRec(on){ dot.style.background = on ? '#e74c3c' : '#bbb'; startBtn.disabled = on; stopBtn.disabled = !on; msg.textContent = on ? 'Grabando…' : 'Listo para grabar'; }

      return await new Promise(async (resolve, reject) => {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio:true });
          rec = new MediaRecorder(stream);
        } catch (e) {
          box.remove(); reject('No mic: ' + e); return;
        }
        rec.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };
        rec.onstop = async () => {
          try {
            const blob = new Blob(chunks, {type:'audio/webm;codecs=opus'});
            const buf = await blob.arrayBuffer();
            const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
            stream.getTracks().forEach(t => t.stop());
            box.remove();
            resolve(b64);
          } catch (e) {
            stream.getTracks().forEach(t => t.stop());
            box.remove();
            reject(e);
          }
        };
        startBtn.onclick = () => { chunks = []; rec.start(); setRec(true); };
        stopBtn.onclick  = () => { if (rec && rec.state === 'recording') { rec.stop(); setRec(false); } };
      });
    }
    """)
    display(js)
    b64 = output.eval_js("recorderUIOnce()")
    webm_tmp = f"/content/rec_{uuid.uuid4().hex}.webm"
    with open(webm_tmp, "wb") as f: f.write(base64.b64decode(b64))
    # Convertir a WAV 16 kHz mono
    subprocess.run(["ffmpeg", "-y", "-i", webm_tmp, "-ac", "1", "-ar", str(sr), out_wav],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try: os.remove(webm_tmp)
    except: pass
    if autoplay: display(Audio(filename=out_wav, autoplay=True))
    return out_wav


Este cambio evita montar múltiples UIs/listeners y desmonta el componente tras parar, eliminando el overflow del call stack. (Tu versión anterior agregaba listeners en cada ejecución y no limpiaba el DOM, que es la causa típica.) Puntos relevantes del original donde ocurría la acumulación de listeners: handler onclick y onstop reinyectados repetidamente.

Indicaciones de implementación (lo que debe generar el modelo con este megaprompt)

ASR (Whisper)

Cargar modelo según VRAM.

Transcribir input.wav → text_transcript.

Medir tiempo: asr_time.

Imprimir ASR: y latencia.

LLM local (FLAN-T5-Base)

AutoTokenizer y AutoModelForSeq2SeqLM + pipeline("text2text-generation").

Prompt corto; respuesta 1–3 oraciones.

Medir llm_time y longitud output.
(Corregir cualquier oTokenizer → AutoTokenizer si aparece por error de exportación.)

TTS (xTTS v2)

Clonado: tts.tts_to_file(text=llm_text, speaker_wav=<ruta_speaker>, language="es", file_path="out_clone.wav")

Base: el mismo text pero sin speaker_wav, p.ej. file_path="out_base.wav"

Medir tts_clone_time, tts_base_time.

Función asr_llm_tts_pipeline

Entradas: input_wav, speaker_wav, lang.

Salida: dict con textos, rutas de audios, tiempos por etapa y total.

Imprimir resumen MECE y reproducir ambos audios.

Entrega (checklist auto-verificable en la última celda)

a) Implementación ASR/LLM/TTS ✔

b) Función integrada ✔

c) Textos y tiempos impresos ✔

Qué recibirá el profesor en demo

Ejecutarán una pregunta en vivo y escucharán la respuesta.

Justificarás elección de modelos, parámetros de latencia/calidad y observaciones del voice cloning vs voz base.
