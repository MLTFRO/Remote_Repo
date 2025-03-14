
def build_loss_vae(args,lambda_reconstruct = .5, lambda_kl = .5):
    def loss_vae(x, x_hat, mean, logvar):
        reconstruct_loss = lambda_reconstruct * (x - x_hat).pow(2).sum()
        KL_loss = -lambda_kl * torch.sum(logvar - mean.pow(2) - logvar.exp())

        return reconstruct_loss + KL_loss
    return loss_vae