import torch
import torch.nn.functional as F

# L2正則化
# 元画像と敵対的サンプルのL2 normがepsilon未満になるように正則化
def project(perturbed_image, original_image, epsilon, _type='l2'):
    if _type == 'l2':
        # 元画像と敵対的サンプルのL2 normを求める
        dist = (perturbed_image - original_image)
        dist = dist.view(perturbed_image.shape[0], -1)
        dist_norm = torch.norm(dist, dim=1, keepdim=True)

        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
        dist = dist / dist_norm
        dist *= epsilon
        dist = dist.view(perturbed_image.shape)

        perturbed_image = (original_image + dist) * mask.float() + perturbed_image * (1 - mask.float())

    else:
        raise NotImplementedError
    
    return perturbed_image

class FastGradientSignUntargeted():
    # model: weight
    def __init__(self, model, epsilon, _type='l2'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = 0.01
        self._type = _type

    def perturb(self, original_images, labels, reduction4loss='mean'):
        original = original_images.clone()
        original.requires_grad = True

        iters = 40
        with torch.enable_grad():
            for _iter in range(iters):
                # vector dim classed
                outputs = self.model(original)

                #print("----------------")
                #print("outputs")
                #print(labels)
                #print(outputs[0].size())   # torch.size([batches, classes])
                #print(labels.size())       # torch.size([batches])

                loss = F.cross_entropy(outputs[0], labels, reduction=reduction4loss)

                if reduction4loss == 'none':
                    grad_outputs = tensor2cuda(torch.ones(loss.shape))
                    
                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, original, grad_outputs=grad_outputs, 
                        only_inputs=True)[0]

                original.data += self.alpha * torch.sign(grads.data)

                original = project(original, original_images, self.epsilon, self._type)
                original.clamp_(0,1)
        return original



class FastGradientSignUntargetedForArchitectureSearch():
    def __init__(self, model, epsilon, _type='l2'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = 0.01
        self._type = _type

    def perturb(self, original_images, labels, reduction4loss='mean'):
        original = original_images.clone()
        original.requires_grad = True

        iters = 40
        with torch.enable_grad():
            for _iter in range(iters):
                outputs = self.model(original)

                #print("----------------")
                #print(outputs.size()) # torch.size([batches, classes])
                #print(labels.size()) # torch.size([batches])

                loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)

                if reduction4loss == 'none':
                    grad_outputs = tensor2cuda(torch.ones(loss.shape))
                    
                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, original, grad_outputs=grad_outputs, 
                        only_inputs=True)[0]

                original.data += self.alpha * torch.sign(grads.data)

                original = project(original, original_images, self.epsilon, self._type)
                original.clamp_(0,1)
        return original