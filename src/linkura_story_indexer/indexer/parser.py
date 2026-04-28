import re

PARSER_VERSION = "1"


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
