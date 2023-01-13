import matplotlib.pyplot as plt

# raw data from sweeping the number of hidden units
# I know this is bad code but it's 1am
data = [0, 0.37331677, 0.34273032, 0.37041467, 0.34437477, 0.32741399,
  0.38801226, 0.41773034, 0.38132376, 0.34489742, 0.32258217, 0.3494152,
  0.39870383, 0.37507409, 0.4070266,  0.34670083, 0.42364214, 0.3676315,
  0.42487701, 0.37480186, 0.32182406, 0.36205609, 0.39293943, 0.40278938,
  0.48068823, 0.38878301, 0.33479451, 0.38100334, 0.36879699, 0.34875111]

plt.plot(data)
plt.xlabel("Hidden Units")
plt.ylabel("F1 Score")
plt.show()
