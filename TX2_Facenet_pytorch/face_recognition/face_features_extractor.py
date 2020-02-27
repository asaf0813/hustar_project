import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from . import preprocessing
from facenet_pytorch.models.utils.detect_face import extract_face

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class FaceFeaturesExtractor:
    def __init__(self):
        self.aligner = MTCNN(prewhiten=False, keep_all=True, thresholds=[0.6, 0.7, 0.9],device=device)
        self.facenet_preprocess = transforms.Compose([preprocessing.Whitening()])
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

    def extract_features(self, img):
        bbs, _ = self.aligner.detect(img)
        if bbs is None:
            # if no face is detected
            return None, None
        faces = []
        for bb in bbs:
            faces.append(extract_face(img, bb))
        faces = torch.stack(faces)
        # faces = torch.stack([extract_face(img, bb) for bb in bbs]).to(device)
        embeddings = self.facenet(self.facenet_preprocess(faces)).detach().numpy()

        return bbs, embeddings

    def __call__(self, img):
        return self.extract_features(img)
