CLASS_NAMES = ["green", "natural", "chemical"]
STAGES = ["GREEN / UNRIPE", "NATURALLY RIPENED", "CHEMICALLY RIPENED"]
RECOMMENDATIONS = {
	"GREEN / UNRIPE": "Allow the banana to ripen naturally before consumption.",
	"NATURALLY RIPENED": "Model predicts natural ripening. This is an image-based estimate.",
	"CHEMICALLY RIPENED": "Model predicts chemical ripening. Further laboratory testing is recommended for confirmation.",
}
DESCRIPTIONS = {
	"GREEN / UNRIPE": "Banana appears to be unripe based on the analyzed image features.",
	"NATURALLY RIPENED": "Banana is predicted as naturally ripened based on the learned image features.",
	"CHEMICALLY RIPENED": "Banana is predicted as chemically ripened based on the learned image features.",
}


def classify_rule_based(features):
	"""Provide an image-based fallback until a real three-class model is trained."""
	green, yellow = features["green"], features["yellow"]
	brown_area = features["brown_area"]
	if green >= 58 and yellow < 25:
		label = 0
	elif brown_area >= 20 and yellow >= 12:
		label = 2
	else:
		label = 1
	stage = STAGES[label]
	return {
		"stage": stage,
		"stage_number": label + 1,
		"confidence": None,
		"recommendation": RECOMMENDATIONS[stage],
		"description": DESCRIPTIONS[stage].replace("learned", "analyzed"),
		"mode": "Image-based classification",
		"probabilities": None,
	}
