import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric

from parse_config import ConfigParser

from utils import calculate_prec_rec_f1, calculate_prec_rec_f1_multibin


def main(config):
    import model.model as module_arch
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['batch_size'],
        config['data_loader']['args']['file'],
        shuffle=False,
        validation_split=0.0,
        num_workers=8
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    loss_fn_multi = module_loss.nll_loss
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    preds_acc = None
    if config.config.get('save_predictions'):
        preds_acc = []

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    total_tp_fp_tn_fn = {'tp':0,'fp':0,'tn':0,'fn':0,'tp_multibin':0,'fp_multibin':0,'tn_multibin':0,'fn_multibin':0}

    with torch.no_grad():
        for i, inputs in enumerate(tqdm(data_loader)):
            data, token_ids, attn_mask, target, target_multi1, target_multi2 = inputs['input_ids'].to(device), inputs['token_type_ids'].to(device), inputs['attention_mask'].to(device), inputs['labels'].to(device), inputs['label_multi1'].to(device), inputs['label_multi2'].to(device)

            target = target.transpose(0, 1)
            target_multi1 = target_multi1.transpose(0, 1).squeeze()
            target_multi2 = target_multi2.transpose(0, 1).squeeze()

            output_binary, output_multi1, output_multi2 = model(data, token_ids, attn_mask)

            multi1_proba = torch.exp(output_multi1)
            _, multi1_pred = multi1_proba.max(1)
            multi2_proba = torch.exp(output_multi2)
            _, multi2_pred = multi2_proba.max(1)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            if config.config.get('pos_neg_ratio'):
                loss_binary = loss_fn(output_binary, target, config['pos_neg_ratio'])
            else:
                loss_binary = loss_fn(output_binary, target)

            loss_multi1 = loss_fn_multi(output_multi1, target_multi1)
            loss_multi2 = loss_fn_multi(output_multi2, target_multi2)

            batch_size = data.shape[0]
            total_loss += (loss_binary + loss_multi1 + loss_multi2).item() * batch_size

            output = output_binary.sigmoid()

            if preds_acc is not None:
                preds_acc.extend(list(output))

            for i, metric in enumerate(metric_fns):
                if 'multi1' in metric.__name__:
                    result = metric(multi1_pred, target_multi1)
                elif 'multi2' in metric.__name__:
                    result = metric(multi2_pred, target_multi2)
                elif 'multibin' in metric.__name__:
                    result = metric(multi1_pred, target_multi1, multi2_pred, target_multi2)
                else:
                    result = metric(output, target)
                if metric.__name__ in list(total_tp_fp_tn_fn.keys()):
                    total_tp_fp_tn_fn[metric.__name__] += result
                total_metrics[i] += result * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    if preds_acc is not None:
        preds_to_write = data_loader.dataset.dataframe
        preds_to_write['predictions'] = preds_acc
        preds_to_write = preds_to_write.loc[:,('pair_id','label','predictions')]
        preds_to_write['predictions'] = preds_to_write['predictions'].apply(lambda x: x.item())
        preds_to_write.to_pickle(f'{config.save_dir}/predictions.pkl.gz')
    precision, recall, f1 = calculate_prec_rec_f1(total_tp_fp_tn_fn)
    precision_multibin, recall_multibin, f1_multibin = calculate_prec_rec_f1_multibin(total_tp_fp_tn_fn)
    additional_log = {"tp": total_tp_fp_tn_fn['tp'], "fp": total_tp_fp_tn_fn['fp'], "tn": total_tp_fp_tn_fn['tn'], "fn": total_tp_fp_tn_fn['fn'], "precision": precision, "recall": recall, "f1": f1,
                      "tp_multibin": total_tp_fp_tn_fn['tp_multibin'], "fp_multibin": total_tp_fp_tn_fn['fp_multibin'], "tn_multibin": total_tp_fp_tn_fn['tn_multibin'], "fn_multibin": total_tp_fp_tn_fn['fn_multibin'], "precision_multibin": precision_multibin, "recall_multibin": recall_multibin, "f1_multibin": f1_multibin}
    log.update(additional_log)

    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
