from pytorch_fid import fid_score
import lpips
import os
from PIL import Image
from tqdm import tqdm 
import threading
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])
def compute_fid(real_images_folder,generated_images_folder,results):
    print("computing fid...")
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                    device="cuda",
                                                    batch_size=1,
                                                    dims=2048)
    fid_value = round(fid_value,3)
    print("fid:",fid_value)
    results["fid"] = fid_value

def compute_lpips(real_images_folder,generated_images_folder,results):
    print("computing lpips...")
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    real_images_name = os.listdir(real_images_folder)
    lpips_alex_total = 0.0

    for filename in tqdm(real_images_name,desc="LPIPS"):
        img1 = Image.open(os.path.join(real_images_folder, filename))
        img2 = Image.open(os.path.join(generated_images_folder,filename))
        img1 = (2.0 * transform(img1) - 1.0).unsqueeze(0).cuda()
        img2 = (2.0 * transform(img2) - 1.0).unsqueeze(0).cuda()
        d1 = loss_fn_alex.forward(img1, img2)
        lpips_alex_total += d1.item()

    lpips_alex_avg = round(lpips_alex_total/len(real_images_name),3)
    print("lpips_alex_avg:",lpips_alex_avg)
    results["lpips_alex_avg"] = lpips_alex_avg

if __name__ == "__main__":
    if os.path.exists('m_class_fid_lpips.txt'):
        os.remove('m_class_fid_lpips.txt')
    classnames = ["bag","top","outwear","pants","dress"]
    # classnames = ["dress"]
    for classname in classnames:
        real_images_folder = r'my_latent_diffusion_big_dataset/test/{}/gt'.format(classname)
        generated_images_folder = r'my_latent_diffusion_big_dataset/test/{}/gen'.format(classname)
        results = {}
        thread1 = threading.Thread(target=compute_fid, args=(real_images_folder,generated_images_folder,results))
        thread2 = threading.Thread(target=compute_lpips, args=(real_images_folder,generated_images_folder,results))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        data = [results["fid"],results["lpips_alex_avg"]]

        # 打开文件并追加写入数据
        with open('m_class_fid_lpips.txt', 'a') as file:
            if classname != classnames[-1]:
                file.write(','.join(map(str, data)) + ',')
            else:
                file.write(','.join(map(str, data)))