def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_train_time_sec = end - start
    hours = total_train_time_sec // 3600
    minutes = (total_train_time_sec % 3600) // 60
    seconds = total_train_time_sec % 60
    total_train_time = f"\nTrain time on {device}: {int(hours)}h {int(minutes)}m {int(seconds)}s"
    return total_train_time_sec,total_train_time