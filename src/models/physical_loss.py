import torch
import torch.nn as nn


class PhysicalLoss(nn.Module):

    def get_loss(self, mask_name, structure):
        valid = torch.tensor([0], dtype=torch.float64, requires_grad=True)
        invalid = torch.tensor([1], dtype=torch.float64, requires_grad=True)

        if mask_name == 'Brainstem':
            return valid if structure.max() <= 54 else invalid
        elif mask_name == 'SpinalCord':
            return valid if structure.max() <= 48 else invalid
        elif mask_name == 'Mandible':
            return valid if structure.max() <= 73.5 else invalid
        elif mask_name == 'RightParotid':
            return valid if structure.mean() <= 26 else invalid
        elif mask_name == 'LeftParotid':
            return valid if structure.mean() <= 26 else invalid
        elif mask_name == 'Larynx':
            return valid if structure.mean() <= 45 else invalid
        elif mask_name == 'Esophagus':
            return valid if structure.mean() <= 45 else invalid
        else:
            raise Exception("%s not valid structure name" % mask_name)

    def forward(self, predicted, batch):
        physical_loss = {}
        for element, mask in zip(predicted, batch['structure_masks']):
            brain_mask = ((mask[..., 0]).flatten() > 0).nonzero().flatten()
            spinal_cord_mask = ((mask[..., 1]).flatten() > 0).nonzero().flatten()
            right_parotid_mask = ((mask[..., 2]).flatten() > 0).nonzero().flatten()
            left_parotid_mask = ((mask[..., 3]).flatten() > 0).nonzero().flatten()
            esophagus_mask = ((mask[..., 4]).flatten() > 0).nonzero().flatten()
            larynx_mask = ((mask[..., 5]).flatten() > 0).nonzero().flatten()
            mandible_mask = ((mask[..., 6]).flatten() > 0).nonzero().flatten()

            if brain_mask.sum() != 0:
                if 'Brainstem' in physical_loss:
                    with torch.no_grad():
                        physical_loss['Brainstem'] += self.get_loss('Brainstem', element.flatten()[brain_mask])
                else:
                    physical_loss['Brainstem'] = self.get_loss('Brainstem', element.flatten()[brain_mask])

            if spinal_cord_mask.sum() != 0:
                if 'SpinalCord' in physical_loss:
                    with torch.no_grad():
                        physical_loss['SpinalCord'] += self.get_loss('SpinalCord', element.flatten()[spinal_cord_mask])
                else:
                    physical_loss['SpinalCord'] = self.get_loss('SpinalCord', element.flatten()[spinal_cord_mask])

            if right_parotid_mask.sum() != 0:
                if 'RightParotid' in physical_loss:
                    with torch.no_grad():
                        physical_loss['RightParotid'] += self.get_loss('RightParotid', element.flatten()[right_parotid_mask])
                else:
                    physical_loss['RightParotid'] = self.get_loss('RightParotid', element.flatten()[right_parotid_mask])

            if left_parotid_mask.sum() != 0:
                if 'LeftParotid' in physical_loss:
                    with torch.no_grad():
                        physical_loss['LeftParotid'] += self.get_loss('LeftParotid', element.flatten()[left_parotid_mask])
                else:
                    physical_loss['LeftParotid'] = self.get_loss('LeftParotid', element.flatten()[left_parotid_mask])

            if esophagus_mask.sum() != 0:
                if 'Esophagus' in physical_loss:
                    with torch.no_grad():
                        physical_loss['Esophagus'] += self.get_loss('Esophagus', element.flatten()[esophagus_mask])
                else:
                    physical_loss['Esophagus'] = self.get_loss('Esophagus', element.flatten()[esophagus_mask])

            if larynx_mask.sum() != 0:
                if 'Larynx' in physical_loss:
                    with torch.no_grad():
                        physical_loss['Larynx'] += self.get_loss('Larynx', element.flatten()[larynx_mask])
                else:
                    physical_loss['Larynx'] = self.get_loss('Larynx', element.flatten()[larynx_mask])

            if mandible_mask.sum() != 0:
                if 'Mandible' in physical_loss:
                    with torch.no_grad():
                        physical_loss['Mandible'] += self.get_loss('Mandible', element.flatten()[mandible_mask])
                else:
                    physical_loss['Mandible'] = self.get_loss('Mandible', element.flatten()[mandible_mask])
        out = (sum(physical_loss.values())/(len(physical_loss))).type_as(predicted)
        return out
