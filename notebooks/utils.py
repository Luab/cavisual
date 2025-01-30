import torch
import numpy as np
import sklearn

def generate_vector(image,target,p=0, ae=None,clf=None):
    image_shape = image.shape
    z = ae.encode(image).detach()
    z.requires_grad = True
    xp = ae.decode(z, image_shape)
    pred = torch.nn.functional.sigmoid(clf((image*p + xp*(1-p))))[:,clf.pathologies.index(target)]
    dzdxp = torch.autograd.grad((pred), z)[0]
    return dzdxp

def egenerate_vector_cav(cav_list, idx, mean=False):
    if mean:
        return torch.mean(torch.stack(cav_list).reshape((-1, 512, 3, 3)).to("cuda").float(),0)
    else: 
        return cav_list[idx].reshape(-1, 512, 3, 3).to("cuda").float()

def generate_explanation(sample, dzdxp, target, p=0, sigma = 0, threshold = False, ae= None, clf=None):
    image = torch.tensor(sample['img']).clone().unsqueeze(0)
    image.requires_grad = True
    image_shape = image.shape
    image = image.to("cuda")
    z = ae.encode(image)
    cache = {}
    fixrange = False
    def compute_shift(lam):
        #print(lam)
        if lam not in cache:
            xpp = ae.decode(z+dzdxp*lam, image_shape)
            pred1 = torch.nn.functional.sigmoid(clf((image*p + xpp*(1-p))))[:,clf.pathologies.index(target)]
            cache[lam] = xpp.detach(), pred1.detach().cpu().numpy()
        return cache[lam]

    #determine range
    #initial_pred = pred.detach().cpu().numpy()
    _, initial_pred = compute_shift(0)


    if fixrange:
        lbound,rbound = fixrange
    else:
        #search params
        step = 10

        #left range
        lbound = 0
        last_pred = initial_pred
        while True:
            xpp, cur_pred = compute_shift(lbound)
            #print("lbound",lbound, "last_pred",last_pred, "cur_pred",cur_pred)
            if last_pred < cur_pred:
                break
            if initial_pred-0.15 > cur_pred:
                break
            if lbound <= -1000:
                break
            last_pred = cur_pred
            if np.abs(lbound) < step:
                lbound = lbound - 1
            else:
                lbound = lbound - step

        #right range
        rbound = 0
        last_pred = initial_pred
        compute_rbound = False

        if compute_rbound:
            while True:
                xpp, cur_pred = compute_shift(rbound)
                #print("rbound",rbound, "last_pred",last_pred, "cur_pred",cur_pred)
                if last_pred > cur_pred:
                    break
                if initial_pred+0.05 < cur_pred:
                    break
                if rbound >= 1000:
                    break
                last_pred = cur_pred
                if np.abs(rbound) < step:
                    rbound = rbound + 1
                else:
                    rbound = rbound + step

    #print(initial_pred, lbound,rbound)
    #lambdas = np.arange(lbound,rbound,(rbound+np.abs(lbound))//10)
    lambdas = np.arange(lbound,rbound,np.abs((lbound-rbound)/10))
    y = []
    dimgs = []
    xp = ae.decode(z,image_shape)[0][0].unsqueeze(0).unsqueeze(0).detach()
    for lam in lambdas:
        xpp, pred = compute_shift(lam)
        dimgs.append(xpp.cpu().numpy())
        y.append(pred)
        
    dimage = np.max(np.abs(xp.cpu().numpy()[0][0] - dimgs[0][0]),0)
    return dimage,dimgs

def thresholdf(x, percentile):
    return x * (x > np.percentile(x, percentile))
    
def calc_iou(preds, gt_seg):
    gt_seg = gt_seg.astype(bool)
    seg_area_percent = (gt_seg > 0).sum()/(gt_seg != -1).sum() # percent of area
    preds = (thresholdf(preds, (1-seg_area_percent)*100) > 0).astype(bool)
    #EPS = 10e-16
    ret = {}
    ret["iou"] = (gt_seg & preds).sum() / ((gt_seg | preds).sum())
    ret["precision"] = sklearn.metrics.precision_score(gt_seg.flatten(),preds.flatten())
    ret["recall"] = sklearn.metrics.recall_score(gt_seg.flatten(),preds.flatten())  
    return ret

def clean(saliency):
    saliency = np.abs(saliency)
    if sigma > 0:
        saliency = skimage.filters.gaussian(saliency, 
                    mode='constant', 
                    sigma=(sigma, sigma), 
                    truncate=3.5)
    if threshold != False:
        saliency = thresholdf(saliency, 95 if threshold == True else threshold)
    return saliency