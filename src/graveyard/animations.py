import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


def jablonski_diagram(time_series, time_step_series, state_series, transition_series, transition_dict, number,
                      state_names, single_states, index_min=0, index_range=100, fps=10, saveas="writer_test.mp4"):
    """
    Animate (part of) the state_series displayed in a Jablonski diagram.

    Parameters
    ----------
    time_series : np.ndarray
        Contains the time points at which the corresponding state occurs.
    time_step_series : np.ndarray
        Contains the time step until the corresponding state occurs (starting from the previous state).
    state_series : np.ndarray
        Contains the consecutive state's unique values.
    transition_series : np.ndarray
        Contains the next transition id for each corresponding state (except the last).
    transition_dict : dict
        Contains transition ids as keys and transition names as values.
    number : int
        Number of fluorophores of the system.
    state_names : Collection
        Contains all state names.
    single_states : iterable object
        Contains elements of type str.
    index_min : int
        Starting index for state_series (and time_series, time_step_series).
    index_range : int
        Number of steps to animate.
    fps : int
        Animation frame rate.
    saveas : str
        Defines the save location of the outfile.

    Returns
    -------
    None
    """
    ffmpegwriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Jablonski diagram", artist="Sagix",
                    comment="Markov Chain visualized in the Jablonski diagram")
    writer = ffmpegwriter(fps=fps, metadata=metadata)

    fig = plt.figure(figsize=(14, 7))
    plt.ylabel("energy", size=21)
    plt.ylim(0, 2)
    plt.xlim(0, 4)
    if single_states == ("S0", "S1", "T1", "B"):
        lines = [0.1, 0.1, 1], [1.5, 0.1, 1], [1.3, 1.1, 2], [0.6, 2.6, 3.5]
        x_diff = 0.9 / number
    elif single_states == ("tS0", "tS1", "tT1", "cS0", "cS1", "B"):
        lines = [0.1, 0.1, 0.6], [1.5, 0.1, 0.6], [1.3, 0.7, 1.2], [0.1, 1.5, 2.0], [1.5, 1.5, 2.0], [0.6, 2.9, 3.4]
        x_diff = 0.5 / number
        plt.vlines(x=1.3, ymin=0, ymax=2, color="k", ls="--")
    elif single_states == ("tS0", "tS1", "tT1", "cS0", "cS1", "OFF", "B"):
        lines = [0.1, 0.1, 0.6], [1.5, 0.1, 0.6], [1.3, 0.7, 1.2], [0.1, 1.5, 2.0], [1.5, 1.5, 2.0], [0.9, 2.9, 3.4], \
                [0.3, 2.9, 3.4]
        x_diff = 0.5 / number
        plt.vlines(x=1.3, ymin=0, ymax=2, color="k", ls="--")
    else:
        raise ValueError
    plt.vlines(x=2.1, ymin=0, ymax=2, color="k", ls="--")
    circle_positions = []
    for line, state in zip(lines, single_states):
        if state not in ("OFF", "B"):
            plt.hlines(y=line[0], xmin=line[1], xmax=line[2], color="k")
        plt.text(x=line[1], y=line[0]+0.1, s=state, size=21)
        circle_position_y = line[0]
        circle_positions_x = np.arange(line[1]+x_diff, line[2]+x_diff, x_diff)
        circle_positions.append([circle_position_y, circle_positions_x])

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    plt.yticks([], [])

    circles = []
    for i in range(number):
        circle, = plt.plot([], [], "ro", markersize=10)
        circles.append(circle)

    exponents = np.floor(np.log10(time_step_series[1:]))
    min_expo = np.min(exponents)

    special_case = False
    next_transition_name = None
    photon = None

    if len(state_series[index_min:]) - 1 <= index_range:
        index_range = len(state_series[index_min:])
        special_case = True

    with writer.saving(fig, saveas, 100):
        for i_1 in range(index_min, index_min + index_range):
            state_index = int(state_series[i_1])
            state = state_names[state_index]
            state = state.split("_")
            for i_2 in range(number):
                index = single_states.index(state[i_2])
                circle_pos = circle_positions[index]
                y = circle_pos[0]
                x = circle_pos[1][i_2]
                if index == len(single_states)-1:
                    circles[i_2].set_color("grey")
                circles[i_2].set_data(x, y)

            if special_case and i_1 == index_range - 1:
                next_transition_in = np.inf
                frames = 1
                next_frame_in = np.inf

            else:
                next_transition_in = time_step_series[i_1 + 1]  # i_1 + 1 because each time interval of state i_1 is
                # the one until it was occupied, here the time it stays at state i (until next transition happens)
                # is desired
                exponent = exponents[i_1]  # i_1 since exponents is defined with array[1:]
                frames = int(1 + exponent - min_expo)
                next_frame_in = frames / fps

            total_time = time_series[i_1]
            latest_transition_name = next_transition_name
            next_transition = transition_series[i_1]
            next_transition_name = transition_dict[next_transition]

            if latest_transition_name:
                if "emission" in latest_transition_name:
                    photon = plt.plot(0.55, 1.8, marker="*", markersize=20, color="r")
                else:
                    if photon:
                        r = photon.pop(0)
                        r.remove()

            row_labels = ["total time", "next transition in", "next frame in", "next transition", "latest transition"]
            cell_texts = [[f"{total_time:.2e} s"], [f"{next_transition_in:.2e} s"], [f"{next_frame_in} s"],
                          [next_transition_name], [latest_transition_name]]
            table = plt.table(cellText=cell_texts, rowLabels=row_labels, cellLoc="center",
                              colWidths=[0.1], rowLoc="center", loc="upper right")
            table.set_fontsize(13)
            table.scale(1.5, 1.5)

            for _ in range(frames):
                writer.grab_frame()

    plt.close()
