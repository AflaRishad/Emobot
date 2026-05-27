def refine_genre(emotion, user_input):

    emotion_map = {
        "happy": ["comedy", "adventure", "fantasy"],
        "sad": ["motivational", "self help", "inspirational"],
        "angry": ["thriller", "action", "psychology"],
        "neutral": ["fiction", "mystery", "romance"]
    }

    possible_genres = emotion_map.get(emotion, ["fiction"])

    for genre in possible_genres:
        if genre in user_input.lower():
            return genre

    return possible_genres[0]