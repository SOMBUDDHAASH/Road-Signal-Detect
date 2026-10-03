"""
Acoustic & Voice ADAS Alert Transducer for Streamlit & Browser Clients.
Maintained by Member D (Integration & Pipeline Lead).

Provides zero-dependency Web Audio synthesize chimes and Web Speech API
voice alerts for critical traffic signs, with temporal deduplication.
"""

from typing import List, Dict, Optional
import time
from src.schema import PipelineDetection, SignCategory


class ADASAudioTransducer:
    """
    Manages audio alert state and generates browser-executable Web Audio
    and Speech Synthesis triggers for real-time driver warnings.
    """

    def __init__(self, cooldown_sec: float = 4.0):
        self.cooldown_sec = cooldown_sec
        self._last_alert_time: Dict[int, float] = {}

    def get_pending_alerts(self, detections: List[PipelineDetection]) -> List[Dict[str, str]]:
        """
        Determines which newly detected signs qualify for an acoustic chime or voice alert.
        """
        now = time.time()
        pending = []

        for det in detections:
            cid = det.classification.class_id
            cname = det.classification.class_name
            cat = det.classification.category
            conf = det.classification.confidence

            # Only alert on high-confidence verified signs
            if cid < 0 or conf < 0.60:
                continue

            last_time = self._last_alert_time.get(cid, 0.0)
            if (now - last_time) >= self.cooldown_sec:
                self._last_alert_time[cid] = now

                # Determine alert type and spoken phrase
                alert_type = "chime_danger"
                spoken_text = f"Warning: {cname} ahead."

                if cid == 14:  # Stop
                    alert_type = "chime_critical"
                    spoken_text = "Stop sign ahead."
                elif cid == 17:  # No Entry
                    alert_type = "chime_critical"
                    spoken_text = "Wrong way. Do not enter."
                elif "Speed limit" in cname:
                    alert_type = "chime_info"
                    speed_num = "".join(filter(str.isdigit, cname))
                    spoken_text = f"Speed limit {speed_num} kilometers per hour."
                elif cat == SignCategory.DANGER:
                    alert_type = "chime_danger"
                    spoken_text = f"Caution: {cname}."
                elif cat == SignCategory.MANDATORY:
                    alert_type = "chime_info"
                    spoken_text = f"{cname}."

                pending.append({
                    "class_id": str(cid),
                    "class_name": cname,
                    "type": alert_type,
                    "speech": spoken_text
                })

        return pending

    def generate_html_audio_payload(
        self,
        detections: List[PipelineDetection],
        enable_sound: bool = True,
        enable_speech: bool = True,
        pending_alerts: Optional[List[Dict[str, str]]] = None
    ) -> Optional[str]:
        """
        Generates an embeddable HTML/JavaScript snippet that executes Web Audio & Speech synthesis.
        """
        if not enable_sound and not enable_speech:
            return None

        pending = pending_alerts if pending_alerts is not None else self.get_pending_alerts(detections)
        if not pending:
            return None

        # Build JS commands for the most urgent alert
        top_alert = pending[0]
        alert_type = top_alert["type"]
        speech_text = top_alert["speech"].replace('"', '\\"')

        # Frequency pairs for tone generator
        freq1, freq2 = (880, 1200) if alert_type == "chime_critical" else ((660, 880) if alert_type == "chime_danger" else (523, 659))

        sound_js = f"""
        try {{
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.type = 'sine';
            osc.frequency.setValueAtTime({freq1}, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime({freq2}, ctx.currentTime + 0.15);
            gain.gain.setValueAtTime(0.2, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.25);
            osc.start();
            osc.stop(ctx.currentTime + 0.25);
        }} catch(e) {{}}
        """ if enable_sound else ""

        speech_js = f"""
        try {{
            if ('speechSynthesis' in window) {{
                const u = new SpeechSynthesisUtterance("{speech_text}");
                u.rate = 1.05;
                u.pitch = 1.0;
                window.speechSynthesis.speak(u);
            }}
        }} catch(e) {{}}
        """ if enable_speech else ""

        html_code = f"""
        <div style="display:none;" id="adas-audio-trigger-{int(time.time()*1000)}">
            <script>
                (function() {{
                    {sound_js}
                    {speech_js}
                }})();
            </script>
        </div>
        """
        return html_code


_TRANSDUCER_INSTANCE: Optional[ADASAudioTransducer] = None


def get_audio_transducer() -> ADASAudioTransducer:
    """Singleton getter for ADASAudioTransducer."""
    global _TRANSDUCER_INSTANCE
    if _TRANSDUCER_INSTANCE is None:
        _TRANSDUCER_INSTANCE = ADASAudioTransducer()
    return _TRANSDUCER_INSTANCE
