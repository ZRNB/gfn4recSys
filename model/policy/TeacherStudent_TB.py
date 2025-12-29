import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from model.general import BaseModel
from model.components import DNN
from model.policy.BaseOnlinePolicy import BaseOnlinePolicy

class TeacherStudent_TB(BaseOnlinePolicy):
    '''
    GFlowNet with Trajectory Balance for listwise recommendation
    
    Original paper for "Trajectory balance: Improved credit assignment in {GF}lowNets"
    @inproceedings{
        malkin2022trajectory,
        title={Trajectory balance: Improved credit assignment in {GF}lowNets},
        author={Nikolay Malkin and Moksh Jain and Emmanuel Bengio and Chen Sun and Yoshua Bengio},
        booktitle={Advances in Neural Information Processing Systems},
        editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
        year={2022},
        url={https://openreview.net/forum?id=5btWTw1vcw1}
    }
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - gfn_forward_hidden_dims
        - gfn_flowzero_hidden_dims
        - gfn_forward_offset
        - gfn_reward_smooth
        - gfn_Z
        - from BaseOnlinePolicy:
            - from BackboneUserEncoder:
                - user_latent_dim
                - item_latent_dim
                - transformer_enc_dim
                - transformer_n_head
                - transformer_d_forward
                - transformer_n_layer
                - state_hidden_dims
                - dropout_rate
                - from BaseModel:
                    - model_path
                    - loss
                    - l2_coef
        '''
        parser = BaseOnlinePolicy.parse_model_args(parser) 
        parser.add_argument('--gfn_forward_hidden_dims', type=int, nargs="+", default=[128], help='hidden dimensions of state_slate encoding layers')
        parser.add_argument('--gfn_flowzero_hidden_dims', type=int, nargs="+", default=[128], help='hidden dimensions of F0')
        parser.add_argument('--gfn_forward_offset', type=float, default=1.0, help='smooth offset of forward logp of TB loss')
        parser.add_argument('--gfn_reward_smooth', type=float, default=1.0, help='reward smooth offset in the backward part of TB loss')
        parser.add_argument('--gfn_Z', type=float, default=0., help='average reward offset')
        parser.add_argument('--gfn_Z_teacher', type=float, default=0.)
        parser.add_argument('--beta_teacher', type=float, default=1.0)
        
        return parser
        
    def __init__(self, args, reader_stats, device):
        # BaseModel initialization: 
        # - reader_stats, model_path, loss_type, l2_coef, no_reg, device, slate_size
        # - _define_params(args)
        self.gfn_forward_hidden_dims = args.gfn_forward_hidden_dims
        self.gfn_flowzero_hidden_dims = args.gfn_flowzero_hidden_dims
        self.gfn_forward_offset = args.gfn_forward_offset
        self.gfn_reward_smooth = args.gfn_reward_smooth
        self.gfn_Z = args.gfn_Z
        self.gfn_Z_teacher = args.gfn_Z_teacher
        self.beta_teacher = args.beta_teacher
        # teacher initialization
        
        super().__init__(args, reader_stats, device)
        self.display_name = "GFN_TS"
        
    def to(self, device):
        new_self = super(TeacherStudent_TB, self).to(device)
        return new_self

    def _define_params(self, args):
        # userEncoder, enc_dim, state_dim, bce_loss
        super()._define_params(args)
        # p_forward
        self.pForwardEncoder = DNN(self.state_dim + self.enc_dim * self.slate_size, 
                                   args.gfn_forward_hidden_dims, self.enc_dim, 
                                   dropout_rate = args.dropout_rate, do_batch_norm = True)
        self.pForwardNorm = nn.LayerNorm(self.enc_dim)
        # flow
        self.logFlowZero = DNN(self.state_dim, args.gfn_flowzero_hidden_dims, 1)

        # teacher model
        self.teacher_pForwardEncoder = DNN(self.state_dim + self.enc_dim * self.slate_size,
                                           args.gfn_forward_hidden_dims, self.enc_dim,
                                           dropout_rate=args.dropout_rate, do_batch_norm=True)
        self.teacher_pForwardNorm = nn.LayerNorm(self.enc_dim)
        self.teacher_logFlowZero = DNN(self.state_dim, args.gfn_flowzero_hidden_dims, 1)
    
    def student_head_parameters(self):
        return (
            list(self.pForwardEncoder.parameters()) +
            list(self.pForwardNorm.parameters()) +
            list(self.logFlowZero.parameters())
        )

    def teacher_head_parameters(self):
        return (
            list(self.teacher_pForwardEncoder.parameters()) +
            list(self.teacher_pForwardNorm.parameters()) +
            list(self.teacher_logFlowZero.parameters())
        )

    def shared_parameters(self):
        # userEncoder + 其他 backbone（如果 BaseOnlinePolicy 里还有）
        return list(self.userEncoder.parameters())
    
    def generate_action(self, user_state, feed_dict):
        candidates = feed_dict['candidates']
        slate_size = feed_dict['action_dim']
        parent_slate = feed_dict['action'] # (B, K)
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        epsilon = feed_dict['epsilon']
        is_teacher = feed_dict['is_teacher']
        '''
        @input:
        - user_state: (B, state_dim) 
        - feed_dict: same as BaseOnlinePolicy.get_forward@feed_dict
        @output:
        - out_dict: {'logP': (B, K), 
                     'logF0': (B,),
                     'action': (B, K), 
                     'reg': scalar}
        '''
        B = user_state.shape[0]
        # batch-wise candidates has shape (B,L), non-batch-wise candidates has shape (1,L)
        batch_wise = True
        if candidates['item_id'].shape[0] == 1:
            batch_wise = False
        # during training, candidates is always the full item set and has shape (1,L) where L=N
        if is_train:
            assert not batch_wise
        # epsilon probability for uniform sampling under exploration
        do_uniform = np.random.random() < epsilon
            
        # (B,)
        logF0 = (self.teacher_logFlowZero if is_teacher else self.logFlowZero)(user_state)
        # (1,L,enc_dim)
        candidate_item_enc, reg = self.userEncoder.get_item_encoding(candidates['item_id'], 
                                                       {k[5:]: v for k,v in candidates.items() if k != 'item_id'}, 
                                                                     B if batch_wise else 1)
        
        # forward probabilities P(a_t|s_{t-1}) of the action, size (B, K)
        current_P = torch.zeros(B, slate_size).to(self.device)
        # log probability, (B, K)
        current_logP = torch.zeros(B, slate_size).to(self.device)
        # (B, K)
        current_action = torch.zeros(B, slate_size).to(torch.long).to(self.device)
        # (B, K, enc_dim)
        current_list_emb = torch.zeros(B, slate_size, self.enc_dim).to(self.device)
        
        forwardNorm = self.teacher_pForwardNorm if is_teacher else self.pForwardNorm
        forwardEncoder = self.teacher_pForwardEncoder if is_teacher else self.pForwardEncoder
        # regressive action generation
        for i in range(slate_size):
            # (B, state_dim + slate_size * enc_dim)
            current_state = torch.cat((user_state.view(B, self.state_dim), current_list_emb.view(B, -1)), dim = 1)
            # (B, enc_dim)
            selection_weight = forwardEncoder(current_state)
            selection_weight = forwardNorm(selection_weight)
            # (B, L)
            score = torch.sum(selection_weight.view(B,1,self.enc_dim) * candidate_item_enc, dim = -1) #/ self.enc_dim
            # (B, L)
            prob = torch.softmax(score, dim = 1)
            logP = torch.log(prob + self.gfn_forward_offset)

            # (B,)
            if is_train or torch.is_tensor(parent_slate):
                # during training, output the target action probability without sampling
                action_at_i = parent_slate[:,i] # (B,)
                current_P[:,i] = torch.gather(prob,1,action_at_i.view(-1,1)).view(-1)
                current_logP[:,i] = torch.gather(logP,1,action_at_i.view(-1,1)).view(-1)
                current_list_emb[:,i,:] = candidate_item_enc.view(-1,self.enc_dim)[action_at_i]
                current_action[:,i] = action_at_i
            else:
                if i > 0:
                    # remove items already selected
                    prob.scatter_(1,current_action[:,:i],0)
                if do_explore:
                    # exploration: categorical sampling or uniform sampling
                    if do_uniform:
                        indices = Categorical(torch.ones_like(prob)).sample()
                    else:
                        indices = Categorical(prob).sample()
                else: 
                    # greedy: topk selection
                    _, indices = torch.topk(prob, k = 1, dim = 1)
                indices = indices.view(-1).detach()
                # update current slate action
                current_action[:,i] = indices
                # update slate action probability
                current_logP[:,i] = torch.gather(logP,1,indices.view(-1,1)).view(-1)
                current_P[:,i] = torch.gather(prob,1,indices.view(-1,1)).view(-1)

                if batch_wise:
                    for j in range(B):
                        current_list_emb[j,i,:] = candidate_item_enc[j,indices[j]]
                else:
                    current_list_emb[:,i,:] = candidate_item_enc.view(-1,self.enc_dim)[indices]
        if is_train and is_teacher == False: 
            ##########################################
            #Need to implement:
            #first version:
            #teacher have not reg in the train phase
            #may be the second version:
            #teacher have the reg in the traiin phase
            ##########################################
            reg = self.get_regularization(self.logFlowZero, forwardEncoder)
        else:
            reg = 0

        out_dict = {'logP': current_logP, 
                    'prob': current_P,
                    'action': current_action, 
                    'logF0': logF0, 
                    'reg': reg}
        return out_dict
    
    def get_loss(self, feed_dict, out_dict):
        '''
        Trajectory balance loss
        @input:
        - feed_dict: same as BaseOnlinePolicy.get_forward@input-feed_dict
        - out_dict: {
            'state': (B,state_dim), 
            'logP': (B,K),
            'logF0': (B,)
            'action': (B,K),
            'reg': scalar, 
            'immediate_response': (B,K*n_feedback),
            'reward': (B,)}
        @output
        - loss
        '''

        #################################
        #need to implement
        #is_teacher == True:
        # require the get_loss of teacher
        #################################
        # (B, )
        is_teacher = feed_dict['is_teacher']
        forward_part = out_dict['logF0'].view(-1) + (self.gfn_Z if not is_teacher else self.gfn_Z_teacher)
        forward_part = forward_part + torch.sum(out_dict['logP'], dim = 1)
        # (B, )
        if is_teacher :
            if not torch.isfinite(feed_dict['new_reward']).all():
                print("NaN/Inf in new_reward before log:", 
                    feed_dict['new_reward'].min().item(), 
                    feed_dict['new_reward'].max().item())
            reward_with_smooth = feed_dict['new_reward'] + self.gfn_reward_smooth
            reward_with_smooth = torch.clamp(reward_with_smooth, min=1e-8)
            backward_part = torch.log(reward_with_smooth).view(-1)
        else:
            backward_part = torch.log(out_dict['reward'] + self.gfn_reward_smooth).view(-1)
        per_sample_TB_loss = (forward_part - backward_part).pow(2)
        # (B, )
        TB_loss = torch.mean((forward_part - backward_part).pow(2))
        # (B, )
        loss = TB_loss + self.l2_coef * out_dict['reg']
        if is_teacher:
            return {'teacher_loss': loss, 'teacher_TB_loss': TB_loss, 'teacher_per_sample_TB_loss': per_sample_TB_loss}
        else:
            return {'loss': loss, 'TB_loss': TB_loss, 'per_sample_TB_loss': per_sample_TB_loss}

        
    def get_loss_observation(self):
        return ['loss', 'TB_loss', 'teacher_loss', 'teacher_TB_loss']
        
        