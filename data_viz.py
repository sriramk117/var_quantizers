from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

def load_scalar(event_file, tag="loss"):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    print(f"Available tags in {event_file}:")
    print(ea.Tags())  # <-- This line shows what scalar tags are actually available

    scalars = ea.Scalars(tag)
    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]
    return steps, values

event_files = {
    "4K FSQ": "logs/events.out.tfevents.1749343059.26115d46775a.2123.0__0608_0837",
    "4K VQVAE": "logs/events.out.tfevents.1749337338.d035ed37969c.22688.0__0608_0702",
    "16K FSQ":  "logs/events.out.tfevents.1749332213.e1f80c46301a.2243.0__0608_0536",
    "64K FSQ": "logs/events.out.tfevents.1749329325.d035ed37969c.7705.0__0608_0448",
}

for label, file in event_files.items():
    if os.path.exists(file):
        print(f"Loading {label} from {file}")
        steps, values = load_scalar(file, tag='AR_ep_loss/vacc_mean')
        plt.plot(steps, values, label=label)

plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.show()
