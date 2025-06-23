import numpy as np


values = [0.0, 1.0, 4.0, 6.0, 13.0, 25.0, 38.0, 39.0, 255.0, 7.0, 8.0, 10.0, 
          12.0, 36.0, 2.0, 9.0, 14.0, 37.0, 3.0, 11.0, 19.0, 21.0, 23.0, 
          26.0, 33.0, 5.0, 15.0, 34.0, 17.0, 24.0, 30.0, 27.0, 32.0, 35.0, 
          28.0, 22.0, 18.0, 16.0, 31.0, 20.0, 29.0]

counts = [52669917, 22390046, 8258474, 4923207, 2424313, 896258, 4564517, 
          11271305, 42998518, 4885861, 5379254, 4864604, 3817783, 510907, 
          16007761, 4680084, 3025242, 5542797, 8578042, 3263526, 1559885, 
          3535427, 1280000, 813915, 589331, 5747842, 2414691, 658110, 
          1942860, 1349733, 737636, 986299, 691338, 529675, 1000698, 
          1308322, 2595688, 2222915, 620795, 2044843, 641581]

ignore_index = None
for i in range(len(values)):
    if values[i] == 255.0:
        ignore_index = i
        break

if ignore_index is not None:
    values.pop(ignore_index)
    counts.pop(ignore_index)


def calculate_inverse_frequency_weights(counts):
    counts = np.array(counts, dtype=np.float32)
    total_pixels = np.sum(counts)
    num_classes = len(counts)
    weights = total_pixels / (num_classes * counts)
    return weights


if __name__ == "__main__":
    print(len(values), len(counts))
    nyuv2_weights = calculate_inverse_frequency_weights(counts)
    print(nyuv2_weights.tolist()) 
    print(len(nyuv2_weights))