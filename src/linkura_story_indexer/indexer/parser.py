import re

from ..models.story import DialogueTurn, NarrativeBeat

PARSER_VERSION = "2"
UNKNOWN_SPEAKER = "UNKNOWN"
_QUOTE_RE = re.compile(r"(「[^」]+」|『[^』]+』|“[^”]+”|\"[^\"]+\")")


class StoryParser:
    """Parses story files into scenes and identifies dialogue/prose."""
    
    @staticmethod
    def split_into_scenes(content: str) -> list[str]:
        """Splits file content by the '---' scene delimiter."""
        # Split and filter out empty segments
        scenes = [s.strip() for s in content.split("---")]
        return [s for s in scenes if s]

    @staticmethod
    def parse_script_line(line: str) -> tuple[str, str]:
        """
        Parses a script line into (speaker, text).
        Matches patterns like 'Kaho: Hello' or '花帆: こんにちは'
        """
        match = re.match(r"^([^:：]+)[:：]\s*(.*)$", line.strip())
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", ""

    @staticmethod
    def detect_speakers(content: str) -> list[str]:
        """Returns unique script speakers in first-seen order."""
        speakers = []
        seen = set()
        for line in content.splitlines():
            speaker, _ = StoryParser.parse_script_line(line)
            if speaker and speaker not in seen:
                speakers.append(speaker)
                seen.add(speaker)
        return speakers

    @staticmethod
    def ordered_unique_speakers(turns: list[DialogueTurn]) -> list[str]:
        """Returns unique speakers in first-seen order."""
        speakers = []
        seen = set()
        for turn in turns:
            if turn.speaker in seen:
                continue
            speakers.append(turn.speaker)
            seen.add(turn.speaker)
        return speakers

    @staticmethod
    def parse_script_scene(scene_text: str, scene_id: str = "") -> list[DialogueTurn]:
        """Parses a script-format scene into ordered dialogue turns."""
        turns = []
        for line_number, line in enumerate(scene_text.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            speaker, text = StoryParser.parse_script_line(stripped)
            if not speaker:
                speaker = UNKNOWN_SPEAKER
                text = stripped
            turn_index = len(turns)
            turn_id = f"turn:{scene_id}:{turn_index}" if scene_id else ""
            turns.append(
                DialogueTurn(
                    turn_id=turn_id,
                    scene_id=scene_id,
                    turn_index=turn_index,
                    speaker=speaker,
                    text=text,
                    line_start=line_number,
                    line_end=line_number,
                )
            )
        return turns

    @staticmethod
    def parse_prose_scene(
        scene_text: str,
        scene_id: str = "",
    ) -> tuple[list[DialogueTurn], list[NarrativeBeat]]:
        """Separates quoted dialogue from narrative beats in a prose-format scene."""
        turns: list[DialogueTurn] = []
        beats: list[NarrativeBeat] = []

        for line_number, line in enumerate(scene_text.splitlines()):
            remaining_start = 0
            for match in _QUOTE_RE.finditer(line):
                before = line[remaining_start : match.start()].strip()
                if before:
                    beat_index = len(beats)
                    beat_id = f"beat:{scene_id}:{beat_index}" if scene_id else ""
                    beats.append(
                        NarrativeBeat(
                            beat_id=beat_id,
                            scene_id=scene_id,
                            beat_index=beat_index,
                            text=before,
                            line_start=line_number,
                            line_end=line_number,
                        )
                    )

                turn_index = len(turns)
                turn_id = f"turn:{scene_id}:{turn_index}" if scene_id else ""
                turns.append(
                    DialogueTurn(
                        turn_id=turn_id,
                        scene_id=scene_id,
                        turn_index=turn_index,
                        speaker=UNKNOWN_SPEAKER,
                        text=match.group(0),
                        line_start=line_number,
                        line_end=line_number,
                    )
                )
                remaining_start = match.end()

            after = line[remaining_start:].strip()
            if after:
                beat_index = len(beats)
                beat_id = f"beat:{scene_id}:{beat_index}" if scene_id else ""
                beats.append(
                    NarrativeBeat(
                        beat_id=beat_id,
                        scene_id=scene_id,
                        beat_index=beat_index,
                        text=after,
                        line_start=line_number,
                        line_end=line_number,
                    )
                )
        return turns, beats

    @staticmethod
    def is_script_format(content: str) -> bool:
        """Heuristically determines if a scene is in script format."""
        lines = content.strip().split("\n")
        # Check first 5 non-empty lines for speaker: text pattern
        script_like_count = 0
        checks = 0
        for line in lines:
            if not line.strip():
                continue
            if StoryParser.parse_script_line(line)[0]:
                script_like_count += 1
            checks += 1
            if checks >= 5:
                break
        
        return script_like_count >= 2  # Simple majority check
