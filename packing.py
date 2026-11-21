"""Minimize the busiest worker's runtime plus two seconds per worker."""

from heapq import heapreplace


def _greedy(jobs, count):
    queue = [(0.0, index) for index in range(count)]
    groups = [[] for _ in queue]
    for duration, test_id in jobs:
        load, worker = queue[0]
        groups[worker].append(test_id)
        heapreplace(queue, (load + duration, worker))
    return max(load for load, _ in queue), groups


def _refine(jobs, count, lower_bound, best_peak, groups, cutoff):
    """Search a bounded tree when there are too few jobs for greedy balancing."""
    loads = [0.0] * count
    bins = [[] for _ in loads]
    budget = 6000 if len(jobs) <= 12 else 1800
    epsilon = 1e-8

    def visit(index, peak):
        nonlocal best_peak, groups, budget
        if budget <= 0 or best_peak <= lower_bound + epsilon or peak >= min(best_peak, cutoff) - epsilon:
            return
        budget -= 1
        if index == len(jobs):
            best_peak = peak
            groups = [group.copy() for group in bins]
            return

        duration, test_id = jobs[index]
        tried = set()
        for worker in sorted(range(count), key=loads.__getitem__):
            load = loads[worker]
            if load + duration >= min(best_peak, cutoff) - epsilon:
                break
            if load in tried:
                continue
            tried.add(load)
            loads[worker] = load + duration
            bins[worker].append(test_id)
            visit(index + 1, max(peak, loads[worker]))
            bins[worker].pop()
            loads[worker] = load

    visit(0, 0.0)
    return best_peak, groups


def pack(test_ids, runtimes):
    """Return groups of integer IDs, given a mapping or sequence of test seconds."""
    jobs = sorted(((runtimes[i], i) for i in set(test_ids)), reverse=True)
    if not jobs:
        return []

    total = sum(duration for duration, _ in jobs)
    largest = jobs[0][0]
    best_cost = total + 2.0
    best_groups = [[test_id for _, test_id in jobs]]
    choices = []
    for count in range(2, min(len(jobs), 8) + 1):
        lower = max(largest, total / count)
        if count < len(jobs):
            # At least two of the largest count+1 jobs share a worker.
            lower = max(lower, jobs[count - 1][0] + jobs[count][0])
        choices.append((lower + 2 * count, count, lower))

    for predicted, count, lower in sorted(choices):
        if predicted >= best_cost - 1e-8:
            break
        peak, groups = _greedy(jobs, count)
        if len(jobs) <= 18 and peak > lower + 0.001:
            peak, groups = _refine(jobs, count, lower, peak, groups, best_cost - 2 * count)
        if peak + 2 * count < best_cost:
            best_groups = groups
            best_cost = peak + 2 * count
    return [group for group in best_groups if group]
