import torch

if __name__ == "__main__":
    model = torch.load('best_val_model')
    example_input = torch.rand(1, 1, 100, 40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(torch.__version__)
    traced_script_module = torch.jit.trace(model, example_input.to(device))
    traced_script_module.save("model.pt")