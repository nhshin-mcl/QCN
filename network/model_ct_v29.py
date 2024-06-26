from network.backbone import BackboneV4
from network.ct import ComparisonTransformerLayer

class CTV29(BackboneV4):
    def __init__(self, cfg):
        super(CTV29, self).__init__(cfg)

        self.kmax_refiner = ComparisonTransformerLayer(
            dec_layers=cfg.dec_layers,
            query_dim=cfg.reduced_dim,
            key_value_dim=cfg.reduced_dim,
            num_queries=cfg.reference_point_num,
            drop_path_prob=0.1
        )

    def forward(self, phase, input_dic):
        if phase == 'extraction':
            f = self._forward(im=input_dic['img'])
            return f

        elif phase == 'get_cluster':
            f, rpt = self.kmax_refiner(input_dic['f'])
            return f, rpt

        else:
            raise ValueError(f'Undefined phase ({phase}) has been given. It should be one of [extraction, get_cluster, get_score].')





