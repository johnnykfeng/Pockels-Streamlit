import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


def radians_to_degrees(radians):
    return radians * (180 / np.pi)


def degrees_to_radians(degrees):
    return degrees * (np.pi / 180)


def malus_law(theta_deg, I0, phi, offset):
    theta_rad = degrees_to_radians(theta_deg)
    transmittance = np.cos(theta_rad + phi) ** 2
    return I0 * transmittance + offset


def fit_malus_law(theta_deg, intensity, phi_guess=0, I0_guess=1, offset_guess=0):
    popt, pcov = curve_fit(
        malus_law, theta_deg, intensity, p0=[phi_guess, I0_guess, offset_guess]
    )
    return popt, pcov


def get_parallel_angle(phi):
    angle1 = 180 - phi
    return angle1


def get_crossed_angle(phi):
    angle2 = 90 - phi
    return angle2


def main():
    thetas = np.arange(0, 180, 10)
    print(f"thetas: \n{thetas}")

    parameters = [1, 1, 0.3]
    I_noisy = malus_law(
        thetas, I0=parameters[0], phi=parameters[1], offset=parameters[2]
    ) + np.random.normal(0, 0.1, len(thetas))
    I_clean = malus_law(
        thetas, I0=parameters[0], phi=parameters[1], offset=parameters[2]
    )
    I_noisy_rounded = np.round(I_noisy, 2)
    print(f"I_noisy_rounded: \n{I_noisy_rounded}")

    plt.plot(thetas, I_noisy, "o", label="data")
    plt.plot(thetas, I_clean, label="clean")
    popt, pcov = fit_malus_law(thetas, I_noisy)
    print(f"popt: {popt}")
    # print(f"pcov: {pcov}")
    plt.plot(thetas, malus_law(thetas, *popt), label="fit", linestyle="--")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
