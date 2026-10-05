# Compute notes

This workspace has CPU access and GPU jobs through Slurm.

```bash
gpu submit -- <command>
gpu status
gpu logs <job-id>
gpu cancel <job-id>
gpu shell <job-id>
```

Each job gets one-eighth of the node: 1 A100, 12 CPUs, and 158 GiB RAM, for up to 12 hours. You may have one running job and up to three submitted jobs. How you spend that compute is yours.

If you train, keep the run small enough to finish in the timebox. If you do not update model weights, say so.

Do not put model weights, adapters, or checkpoints in the submission. If an artifact matters to your argument, describe it in the writeup.
