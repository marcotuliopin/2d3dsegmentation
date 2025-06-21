import numpy as np


counts = np.array([52669917, 22390046, 8578042, 4885861, 4864604, 2414691, 1942860, 2044843, 
                   813915, 658110, 510907, 11271305, 16007761, 3263526, 737636, 
                   4564517, 8258474, 3817783, 2424313, 3025242, 896258, 641581, 1308322, 
                   1349733, 1000698, 620795, 4923207, 5542797, 3535427, 1280000, 5379254, 
                   589331, 2595688, 986299, 529675, 5747842, 691338, 4680084, 1559885, 2222915])

def calculate_inverse_frequency_weights(counts):
    total_pixels = np.sum(counts)
    num_classes = len(counts)
    weights = total_pixels / (num_classes * counts)
    return weights


if __name__ == "__main__":
    nyuv2_weights = calculate_inverse_frequency_weights(counts)
    print("NYUv2 Weights (13 classes):")
    print(np.sqrt(nyuv2_weights))