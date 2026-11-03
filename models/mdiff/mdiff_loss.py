# Local adaptations for the CNN–ViT LRLPR study: imports, common input/vocabulary,
# and native-model integration; implementation retained from the validated study.
# Upstream authors, applicable license texts and provenance are listed in NOTICE
# and SOURCE_PROVENANCE_LEDGER.csv. Original copyright notices below are retained.
from torch import nn


class MDiffLoss(nn.Module):

    def __init__(self, **kwargs):
        super(MDiffLoss, self).__init__()

    def forward(self, predicts, batch):

        return {'loss': predicts}
