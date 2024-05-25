# KNN-MT-PG

### Setup (Linux and Mac OS only)

1. To be able to use the `runpy` command to start/stop the database and perform other basic actions, you'll need to add the following to your .bashrc and re-enter the directory either by restarting the terminal or by issuing a command like `cd ../knn-mt` (assuming you have named the project directory `/knn-mt`):

    ```shell
    run_fridayrc() {
        if [[ -f .fridayrc ]];
        then
            source .fridayrc;
            echo "Using current directory .fridayrc";
        fi
    }

    cd() {
        command cd "$@" &&
        if [[ -f .fridayrc ]];
        then
            run_fridayrc;
        fi
    }

    run_fridayrc
    ```

2. For any of the above commands to work, you will need to have a local `docker` installation running with some kind of VM. The choice is yours, but the author prefers [colima](https://github.com/abiosoft/colima) which works on both Mac OS and Linux.
    > Note: Use `brew install colima` on both Mac OS and Linux. At the time of writing (April 2024), there is no easier and more straightforward way to get `colima` up and running.

### Start/stop the database:

-   Start: `$ runpy start-db`
-   Stop: `$ runpy stop-db`

### Start a psql session in the postgres DB:

-   `runpy psql`

### Usage notes

1. If `.fridayrc` is not being applied properly, just use `python run.py` in place of every instance of `runpy` above to achieve the same result.
