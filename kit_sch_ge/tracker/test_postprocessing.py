"""Some minimal examples to eval the untangling step of the post-processing."""
from unittest.mock import Mock
from tracker.postprocessing import setup_constraints, solve_untangling_problem


def test_merge_case_1():
    #   / 2 \
    # 1       4 - 4
    #   \ 3 /
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3))}
    track_1.successors = {2, 3}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [1]
    track_2.masks = {2: ((1), (2))}
    track_2.successors = {4}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [1]
    track_3.masks = {2: ((1), (3))}
    track_3.successors = {4}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [2, 3]
    track_4.masks = {3: ((1, 1), (2, 3)), 4: ((1, 1), (2, 3))}

    track_4.successors = {}
    all_tracks[4] = track_4

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'m_2_3': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_merge_case_2():
    #   / 2 \   / 5 \
    # 1       4      7 - 7
    #   \ 3 /   \ 6 /
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3))}
    track_1.successors = {2, 3}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [1]
    track_2.masks = {2: ((1), (2))}
    track_2.successors = {4}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [1]
    track_3.masks = {2: ((1), (3))}
    track_3.successors = {4}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [2, 3]
    track_4.masks = {3: ((1, 1), (2, 3))}
    track_4.successors = {5, 6}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [4]
    track_5.masks = {4: ((1), (2))}
    track_5.successors = {7}
    all_tracks[5] = track_5

    track_6 = Mock()
    track_6.track_id = 6
    track_6.pred_track_id = [4]
    track_6.masks = {4: ((1), (3))}
    track_6.successors = {7}
    all_tracks[6] = track_6

    track_7 = Mock()
    track_7.track_id = 6
    track_7.pred_track_id = [5, 6]
    track_7.masks = {5: ((1, 1), (2, 3)), 6: ((1, 1), (2, 3))}
    track_7.successors = {}
    all_tracks[7] = track_7

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'m_2_3': 1, 'm_5_6': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_merge_case_3():
    #         / 2 \
    # 1 -1 -1       4 - 4 - 4
    #         \ 3 /
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3)), 3: ((1, 1), (2, 3))}
    track_1.successors = {2, 3}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [1]
    track_2.masks = {4: ((1), (2))}
    track_2.successors = {4}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [1]
    track_3.masks = {4: ((1), (3))}
    track_3.successors = {4}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [2, 3]
    track_4.masks = {5: ((1, 1), (2, 3)), 6: ((1, 1), (2, 3)), 7: ((1, 1), (2, 3))}
    track_4.successors = {}
    all_tracks[4] = track_4

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'m_2_3': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_merge_case_4():
    #       / 1 \           / 4
    # 6 - 6       3 - 3 - 3
    #       \ 2 /           \ 5
    all_tracks = dict()

    track_6 = Mock()
    track_6.track_id = 6
    track_6.pred_track_id = [0]
    track_6.masks = {0: ((1, 1), (2, 3))}
    track_6.successors = {1, 2}
    all_tracks[6] = track_6

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [6]
    track_1.masks = {1: ((1, 1), (2, 3))}
    track_1.successors = {3}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [6]
    track_2.masks = {1: ((1), (4))}
    track_2.successors = {3}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [1, 2]
    track_3.masks = {2: ((1), (3)), 3: ((1), (3)), 4: ((1), (3))}
    track_3.successors = {4, 5}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [3]
    track_4.masks = {5: ((1, 1), (2, 3))}
    track_4.successors = {}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [3]
    track_5.masks = {5: ((1, 1), (4, 5))}
    track_5.successors = {}
    all_tracks[5] = track_5

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'m_1_2': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_merge_case_5():
    #          / 2 \
    # 1 -1 -1 -  5  - 4 - 4 - 4
    #          \ 3 /
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3)), 3: ((1, 1), (2, 3))}
    track_1.successors = {2, 3, 5}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [1]
    track_2.masks = {4: ((1), (2))}
    track_2.successors = {4}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [1]
    track_3.masks = {4: ((1), (3))}
    track_3.successors = {4}
    all_tracks[3] = track_3

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [1]
    track_5.masks = {4: ((1), (4))}
    track_5.successors = {4}
    all_tracks[5] = track_5

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [2, 3, 5]
    track_4.masks = {5: ((1, 1), (2, 3)), 6: ((1, 1), (2, 3)), 7: ((1, 1), (2, 3))}
    track_4.successors = {}
    all_tracks[4] = track_4

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'m_2_3_5': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_merge_case_6():
    #          / 2
    # 1 -1 -1  - 3  - 5 - 5 - 5
    #            4 /
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3)), 3: ((1, 1), (2, 3))}
    track_1.successors = {2, 3}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [1]
    track_2.masks = {4: ((1), (2))}
    track_2.successors = {}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [1]
    track_3.masks = {4: ((1), (3))}
    track_3.successors = {5}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [0]
    track_4.masks = {4: ((1), (2))}
    track_4.successors = {5}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [3, 4]
    track_5.masks = {5: ((1, 1), (2, 3)), 6: ((1, 1), (2, 3)), 7: ((1, 1), (2, 3))}
    track_5.successors = {}
    all_tracks[5] = track_5

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'m_3_4': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_merge_case_7():
    #   / 2 \   / 5 \
    # 1       4 - 8  - 7 - 7
    #   \ 3 /   \ 6 /
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3))}
    track_1.successors = {2, 3}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [1]
    track_2.masks = {2: ((1), (2))}
    track_2.successors = {4}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [1]
    track_3.masks = {2: ((1), (3))}
    track_3.successors = {4}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [2, 3]
    track_4.masks = {3: ((1, 1), (2, 3))}
    track_4.successors = {5, 6, 8}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [4]
    track_5.masks = {4: ((1), (2))}
    track_5.successors = {7}
    all_tracks[5] = track_5

    track_6 = Mock()
    track_6.track_id = 6
    track_6.pred_track_id = [4]
    track_6.masks = {4: ((1), (3))}
    track_6.successors = {7}
    all_tracks[6] = track_6

    track_8 = Mock()
    track_8.track_id = 8
    track_8.pred_track_id = [4]
    track_8.masks = {4: ((1, 1), (6, 7))}
    track_8.successors = {7}
    all_tracks[8] = track_8

    track_7 = Mock()
    track_7.track_id = 6
    track_7.pred_track_id = [5, 6, 8]
    track_7.masks = {5: ((1, 1), (2, 3)), 6: ((1, 1), (2, 3))}
    track_7.successors = {}
    all_tracks[7] = track_7

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'m_2_3': 1, 'm_5_6_8': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_merge_case_8():
    #       / 3
    # 1 - 2 - 4
    #       \ 5 - 6
    #           \ 7
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3))}
    track_1.successors = {2}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [1]
    track_2.masks = {2: ((1), (2))}
    track_2.successors = {3, 4, 5}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [2]
    track_3.masks = {3: ((1), (3))}
    track_3.successors = {}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [2]
    track_4.masks = {3: ((1), (2))}
    track_4.successors = {}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [2]
    track_5.masks = {3: ((1), (1))}
    track_5.successors = {6, 7}
    all_tracks[5] = track_5

    track_6 = Mock()
    track_6.track_id = 6
    track_6.pred_track_id = [5]
    track_6.masks = {4: ((1), (3))}
    track_6.successors = {}
    all_tracks[6] = track_6

    track_7 = Mock()
    track_7.track_id = 7
    track_7.pred_track_id = [5]
    track_7.masks = {4: ((1), (2))}
    track_7.successors = {}
    all_tracks[7] = track_7

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'m_3_4': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_merge_case_9():
    #        3 \
    #      /      8
    # 1 - 2 - 4/
    #       \ 5 - 6
    #           \ 7
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3))}
    track_1.successors = {2}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [1]
    track_2.masks = {2: ((1), (2))}
    track_2.successors = {3, 4, 5}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [2]
    track_3.masks = {3: ((1), (3))}
    track_3.successors = {8}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [2]
    track_4.masks = {3: ((1), (2))}
    track_4.successors = {8}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [2]
    track_5.masks = {3: ((1), (1))}
    track_5.successors = {6, 7}
    all_tracks[5] = track_5

    track_6 = Mock()
    track_6.track_id = 6
    track_6.pred_track_id = [5]
    track_6.masks = {4: ((1), (3))}
    track_6.successors = {}
    all_tracks[6] = track_6

    track_7 = Mock()
    track_7.track_id = 7
    track_7.pred_track_id = [5]
    track_7.masks = {4: ((1), (2))}
    track_7.successors = {}
    all_tracks[7] = track_7

    track_8 = Mock()
    track_8.track_id = 8
    track_8.pred_track_id = [3, 4]
    track_8.masks = {4: ((1), (2))}
    track_8.successors = {}
    all_tracks[8] = track_8

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'m_3_4': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_split_case_1():
    #        / 1 - 1 \   / 4 - 4
    # 6 - 6            3
    #        \ 2 - 2 /   \ 5 - 5
    all_tracks = dict()

    track_6 = Mock()
    track_6.track_id = 1
    track_6.pred_track_id = [0]
    track_6.masks = {0: ((1, 1), (2, 3))}
    track_6.successors = {1,2}
    all_tracks[6] = track_6

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [6]
    track_1.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3))}
    track_1.successors = {3}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [6]
    track_2.masks = {1: ((1, 1), (4, 5)), 2: ((1, 1), (4, 5))}
    track_2.successors = {3}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [1, 2]
    track_3.masks = {3: ((1, 1, 1, 1), (2, 3, 4, 5))}
    track_3.successors = {4, 5}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [3]
    track_4.masks = {4: ((1, 1), (2, 3)), 5: ((1, 1), (2, 3))}
    track_4.successors = {}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [3]
    track_5.masks = {4: ((1, 1), (4, 5)), 5: ((1, 1), (4, 5))}
    track_5.successors = {}
    all_tracks[5] = track_5

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'s_3': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_split_case_2():
    # 1 \   / 5
    # 2 - 4 - 6
    # 3 /   \ 7
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3))}
    track_1.successors = {4}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [0]
    track_2.masks = {1: ((1, 1), (4, 5))}
    track_2.successors = {4}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [0]
    track_3.masks = {1: ((1, 1), (6, 7))}
    track_3.successors = {4}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [1, 2, 3]
    track_4.masks = {2: ((1, 1, 1, 1, 1, 1), (2, 3, 4, 5, 6, 7))}
    track_4.successors = {5, 6, 7}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [4]
    track_5.masks = {3: ((1, 1), (2, 3))}
    track_5.successors = {}
    all_tracks[5] = track_5

    track_6 = Mock()
    track_6.track_id = 6
    track_6.pred_track_id = [4]
    track_6.masks = {3: ((1, 1), (4, 5))}
    track_6.successors = {}
    all_tracks[6] = track_6

    track_7 = Mock()
    track_7.track_id = 7
    track_7.pred_track_id = [4]
    track_7.masks = {3: ((1, 1), (6, 7))}
    track_7.successors = {}
    all_tracks[7] = track_7

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'s_4': 2}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_split_case_3():
    # 1 - 1 \
    #         3
    # 2 - 2 /
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3))}
    track_1.successors = {3}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [0]
    track_2.masks = {1: ((1, 1), (4, 5)), 2: ((1, 1), (4, 5))}
    track_2.successors = {3}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [1, 2]
    track_3.masks = {3: ((1, 1, 1, 1), (2, 3, 4, 5))}
    track_3.successors = {}
    all_tracks[3] = track_3

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'s_3': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_split_case_4():
    # 1 \   / 5 -5 -5
    #     4 - 6 - 6
    # 2 /   \ 7 - 7
    #       \ 8 - 8
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3))}
    track_1.successors = {4}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [0]
    track_2.masks = {1: ((1, 1), (4, 5))}
    track_2.successors = {4}
    all_tracks[2] = track_2

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [1, 2]
    track_4.masks = {2: ((1, 1, 1, 1, 1, 1), (2, 3, 4, 5, 6, 7))}
    track_4.successors = {5, 6, 7, 8}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [4]
    track_5.masks = {3: ((1, 1), (2, 3)), 4: ((1, 1), (2, 3)), 5: ((1, 1), (2, 3))}
    track_5.successors = {}
    all_tracks[5] = track_5

    track_6 = Mock()
    track_6.track_id = 6
    track_6.pred_track_id = [4]
    track_6.masks = {3: ((1, 1), (4, 5)), 4: ((1, 1), (4, 5))}
    track_6.successors = {}
    all_tracks[6] = track_6

    track_7 = Mock()
    track_7.track_id = 7
    track_7.pred_track_id = [4]
    track_7.masks = {3: ((1, 1), (6, 7)), 4: ((1, 1), (6, 7))}
    track_7.successors = {}
    all_tracks[7] = track_7

    track_8 = Mock()
    track_8.track_id = 8
    track_8.pred_track_id = [4]
    track_8.masks = {3: ((1, 1), (8, 9)), 4: ((1, 1), (8, 9))}
    track_8.successors = {}
    all_tracks[8] = track_8

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'s_4': 2}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_split_case_5():
    # 1 - 1 - 1 - 1 \  / 6 - 6 - 6
    #                5
    #       / 3 - 3 /  \ 7 - 7 - 7
    # 2 - 2
    #       \ 4 - 4
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3)), 3: ((1, 1), (2, 3)), 4: ((1, 1), (2, 3))}
    track_1.successors = {5}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [0]
    track_2.masks = {1: ((1, 1), (4, 5)), 2: ((1, 1), (4, 5))}
    track_2.successors = {3, 4}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [2]
    track_3.masks = {3: ((1, 1, 1, 1), (2, 3, 4, 5)), 4: ((1, 1, 1, 1), (2, 3, 4, 5))}
    track_3.successors = {5}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [2]
    track_4.masks = {3: ((1, 1, 1, 1), (2, 3, 4, 5)), 4: ((1, 1, 1, 1), (6, 7, 8, 9))}
    track_4.successors = {}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [1, 3]
    track_5.masks = {5: ((1, 1, 1, 1), (2, 3, 4, 5))}
    track_5.successors = {6, 7}
    all_tracks[5] = track_5

    track_6 = Mock()
    track_6.track_id = 6
    track_6.pred_track_id = [5]
    track_6.masks = {6: ((1, 1), (2, 3)), 7: ((1, 1), (2, 3))}
    track_6.successors = {}
    all_tracks[6] = track_6

    track_7 = Mock()
    track_7.track_id = 7
    track_7.pred_track_id = [5]
    track_7.masks = {6: ((1, 1), (4, 5)), 7: ((1, 1), (4, 5))}
    track_7.successors = {}
    all_tracks[7] = track_7

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'s_5': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_split_case_6():
    # 1 - 1 \   / 5 - 5
    #         4
    # 2 - 2 /   \ 9 \   / 7 - 7
    #                 6
    # 3 - 3 - 3 - 3 /    \ 8 - 8
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3))}
    track_1.successors = {4}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [0]
    track_2.masks = {1: ((1, 1), (4, 5)), 2: ((1, 1), (4, 5))}
    track_2.successors = {4}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [0]
    track_3.masks = {1: ((1, 1), (6, 7)), 2: ((1, 1), (6, 7)), 3: ((1, 1), (6, 7)), 4: ((1, 1), (6, 7))}
    track_3.successors = {6}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [1, 2]
    track_4.masks = {3: ((1, 1, 1, 1), (2, 3, 4, 5))}
    track_4.successors = {5, 9}
    all_tracks[4] = track_4

    track_9 = Mock()
    track_9.track_id = 9
    track_9.pred_track_id = [4]
    track_9.masks = {4: ((1, 1), (2, 3))}
    track_9.successors = {6}
    all_tracks[9] = track_9

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [4]
    track_5.masks = {4: ((1, 1), (2, 3))}
    track_5.successors = {}
    all_tracks[5] = track_5

    track_6 = Mock()
    track_6.track_id = 6
    track_6.pred_track_id = [3, 9]
    track_6.masks = {5: ((1, 1), (5, 6))}
    track_6.successors = {7, 8}
    all_tracks[6] = track_6

    track_7 = Mock()
    track_7.track_id = 7
    track_7.pred_track_id = [6]
    track_7.masks = {6: ((1, 1), (4, 5)), 7: ((1, 1), (4, 5))}
    track_7.successors = {}
    all_tracks[7] = track_7

    track_8 = Mock()
    track_8.track_id = 8
    track_8.pred_track_id = [6]
    track_8.masks = {6: ((1, 1), (2, 3)), 7: ((1, 1), (2, 3))}
    track_8.successors = {}
    all_tracks[8] = track_8

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'s_4': 1, 's_6': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_split_case_7():
    # 1 - 1 \   / 5
    #         4
    # 2 - 2 /   \
    #             6 \
    #                 7
    # 3 - 3 - 3 - 3 /
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3))}
    track_1.successors = {4}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [0]
    track_2.masks = {1: ((1, 1), (4, 5)), 2: ((1, 1), (4, 5))}
    track_2.successors = {4}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [0]
    track_3.masks = {1: ((1, 1), (6, 7)), 2: ((1, 1), (6, 7)), 3: ((1, 1), (6, 7)), 4: ((1, 1), (6, 7))}
    track_3.successors = {7}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [1, 2]
    track_4.masks = {3: ((1, 1, 1, 1), (2, 3, 4, 5))}
    track_4.successors = {5, 6}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [4]
    track_5.masks = {4: ((1, 1), (2, 3))}
    track_5.successors = {}
    all_tracks[5] = track_5

    track_6 = Mock()
    track_6.track_id = 6
    track_6.pred_track_id = [4]
    track_6.masks = {4: ((1, 1), (5, 6))}
    track_6.successors = {7}
    all_tracks[6] = track_6

    track_7 = Mock()
    track_7.track_id = 7
    track_7.pred_track_id = [3, 6]
    track_7.masks = {5: ((1, 1), (5, 6))}
    track_7.successors = {}
    all_tracks[7] = track_7

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'s_4': 1, 's_7': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_split_merge_case():
    #             / 1 - 1 \   /     4     \
    # 10-  10 - 10          3               - 9 - 9 - 9 - 9
    #             \ 2 - 2 /   \ 5 - 6 - 7 /
    #                             \ 8 /
    all_tracks = dict()

    track_10 = Mock()
    track_10.track_id = 10
    track_10.pred_track_id = [0]
    track_10.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3)), 3: ((1, 1), (2, 3))}
    track_10.successors = {1, 2}
    all_tracks[10] = track_10

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [10]
    track_1.masks = {4: ((1, 1), (2, 3)), 5: ((1, 1), (2, 3))}
    track_1.successors = {3}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [10]
    track_2.masks = {4: ((1, 1), (4, 5)), 5: ((1, 1), (4, 5))}
    track_2.successors = {3}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [1, 2]
    track_3.masks = {6: ((1, 1, 1, 1), (2, 3, 4, 5))}
    track_3.successors = {4, 5}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [3]
    track_4.masks = {7: ((1, 1), (4, 5))}
    track_4.successors = {9}
    all_tracks[4] = track_4

    track_5 = Mock()
    track_5.track_id = 5
    track_5.pred_track_id = [3]
    track_5.masks = {7: ((1, 1), (2, 3))}
    track_5.successors = {6, 8}
    all_tracks[5] = track_5

    track_6 = Mock()
    track_6.track_id = 6
    track_6.pred_track_id = [5]
    track_6.masks = {8: ((1), (2))}
    track_6.successors = {7}
    all_tracks[6] = track_6

    track_8 = Mock()
    track_8.track_id = 8
    track_8.pred_track_id = [5]
    track_8.masks = {8: ((1), (3))}
    track_8.successors = {7}
    all_tracks[8] = track_8

    track_7 = Mock()
    track_7.track_id = 7
    track_7.pred_track_id = [6, 8]
    track_7.masks = {9: ((1, 1), (2, 3))}
    track_7.successors = {9}
    all_tracks[7] = track_7

    track_9 = Mock()
    track_9.track_id = 9
    track_9.pred_track_id = [4, 7]
    track_9.masks = {10: ((1, 1), (2, 3)), 11: ((1, 1), (2, 3)), 12: ((1, 1), (2, 3)), 13: ((1, 1), (2, 3))}
    track_9.successors = {}
    all_tracks[9] = track_9

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {'s_3': 1, 'm_6_8': 1, 'e_4_9': 1}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_no_error_1():
    # 1 - 1
    #
    # 2 - 2 - 3
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3))}
    track_1.successors = {}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [0]
    track_2.masks = {1: ((1, 1), (4, 5)), 2: ((1, 1), (4, 5))}
    track_2.successors = {3}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [2]
    track_3.masks = {3: ((1, 1, 1, 1), (2, 3, 4, 5))}
    track_3.successors = {}
    all_tracks[3] = track_3

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_no_error_2():
    # 1 - 1
    #
    # 2 - 2 - 3
    #        \ 4
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3))}
    track_1.successors = {}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [0]
    track_2.masks = {1: ((1, 1), (4, 5)), 2: ((1, 1), (4, 5))}
    track_2.successors = {3, 4}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [2]
    track_3.masks = {3: ((1, 1, 1, 1), (2, 3, 4, 5))}
    track_3.successors = {}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [2]
    track_4.masks = {3: ((1, 1, 1, 1), (6, 7, 8, 9))}
    track_4.successors = {}
    all_tracks[4] = track_4

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


def test_no_error_3():
    # 1 - 2 - 3
    #        \ 4
    all_tracks = dict()

    track_1 = Mock()
    track_1.track_id = 1
    track_1.pred_track_id = [0]
    track_1.masks = {1: ((1, 1), (2, 3)), 2: ((1, 1), (2, 3))}
    track_1.successors = {2}
    all_tracks[1] = track_1

    track_2 = Mock()
    track_2.track_id = 2
    track_2.pred_track_id = [1]
    track_2.masks = {3: ((1, 1), (4, 5))}
    track_2.successors = {3, 4}
    all_tracks[2] = track_2

    track_3 = Mock()
    track_3.track_id = 3
    track_3.pred_track_id = [2]
    track_3.masks = {4: ((1, 1, 1, 1), (2, 3, 4, 5))}
    track_3.successors = {}
    all_tracks[3] = track_3

    track_4 = Mock()
    track_4.track_id = 4
    track_4.pred_track_id = [2]
    track_4.masks = {4: ((1, 1, 1, 1), (6, 7, 8, 9))}
    track_4.successors = {}
    all_tracks[4] = track_4

    out = setup_constraints(all_tracks)
    result = solve_untangling_problem(*out)

    non_zero_variables = {k: v for k, v in result.items() if v > 0}

    solution = {}
    assert non_zero_variables == solution, f'optimization result {non_zero_variables} and solution {solution} missmatch'


if __name__ == '__main__':
    test_merge_case_8()



