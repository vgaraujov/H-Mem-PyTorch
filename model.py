from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class HMemQA(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(HMemQA, self).__init__()
        self.memory_size = config.memory_size
        self.hops = config.hops
        self.encode = InputModule(config)
        self.bn_story = nn.BatchNorm1d(config.embedding_size)
        self.bn_query = nn.BatchNorm1d(config.embedding_size)
        self.extract = ExtractionModule(config)
        self.write = WritingCell(config)
        self.read = ReadingCell(config)
        self.output = OutputModule(config)

    def init_hidden(self, batch_size, device = None):
        if device: return torch.zeros(batch_size, self.memory_size).to(device)
        else: return torch.zeros(batch_size, self.memory_size)

    def init_memory(self, batch_size, device = None):
        if device: return torch.zeros(batch_size, self.memory_size, self.memory_size).to(device)
        else: return torch.zeros(batch_size, self.memory_size, self.memory_size)

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        batch_size, story_length, _ = story.shape
        device = story.device

        story_emb, query_emb = self.encode(story, query)
        story_emb = self.bn_story(story_emb.permute(0,2,1))
        story_emb = story_emb.permute(0,2,1)
        query_emb = self.bn_query(query_emb)

        k, v = self.extract(story_emb)

        m = self.init_memory(batch_size, device)
        for i in range(story_length):
            m = self.write(k[:,i], v[:,i], m)

        v_r = self.init_hidden(batch_size, device)
        for i in range(self.hops):
            v_r = self.read(query_emb, m, v_r)
        
        logits = self.output(v_r)
        
        return logits

class InputModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InputModule, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=config.vocab_size,
                                       embedding_dim=config.embedding_size,
                                       padding_idx=0)
        nn.init.kaiming_uniform_(self.word_embed.weight, mode='fan_in', nonlinearity='relu')
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.ones(config.max_seq, config.embedding_size))
        nn.init.ones_(self.pos_embed.data)
        self.pos_embed.data /= config.max_seq

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        # Sentence embedding
        sentence_embed = self.word_embed(story)  # [b, s, w, e]
        sentence_sum = torch.einsum('bswe,we->bse', sentence_embed, self.pos_embed[:sentence_embed.shape[2]])
        # Query embedding
        query_embed = self.word_embed(query)  # [b, w, e]
        query_sum = torch.einsum('bwe,we->be', query_embed, self.pos_embed[:query_embed.shape[1]])
        return sentence_sum, query_sum
    
class ExtractionModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(ExtractionModule, self).__init__()
        self.projector_k = nn.Linear(in_features=config.embedding_size, 
                                     out_features=config.memory_size, 
                                     bias=config.use_bias)
        nn.init.kaiming_uniform_(self.projector_k.weight, mode='fan_in', nonlinearity='relu')
        self.projector_v = nn.Linear(in_features=config.embedding_size, 
                                     out_features=config.memory_size, 
                                     bias=config.use_bias)
        nn.init.kaiming_uniform_(self.projector_v.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, sentence: torch.Tensor):
        k = F.relu(self.projector_k(sentence))
        v = F.relu(self.projector_v(sentence))
        return k, v

class OutputModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(OutputModule, self).__init__()
        self.output = nn.Linear(in_features=config.memory_size, 
                                out_features=config.vocab_size, 
                                bias=config.use_bias)
        nn.init.kaiming_uniform_(self.output.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, v_r: torch.Tensor):
        logits = self.output(v_r)
        return logits
    
class WritingCell(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(WritingCell, self).__init__()
        self.read_before_write = config.read_before_write
        self.w_max = config.w_assoc_max
        self.gamma_pos = config.gamma_pos
        self.gamma_neg = config.gamma_neg
        self.projector = nn.Linear(in_features=config.memory_size+config.memory_size, 
                               out_features=config.memory_size, 
                               bias=config.use_bias)
        nn.init.kaiming_uniform_(self.projector.weight, mode='fan_in', nonlinearity='relu')
        
        if self.read_before_write:
            self.ln1 = nn.LayerNorm(config.memory_size)
            self.ln2 = nn.LayerNorm(config.memory_size)

    def forward(self, k: torch.Tensor, v: torch.Tensor, states: torch.Tensor):
        memory_matrix = states
        
        if self.read_before_write:
            k = self.ln1(k)
            v_h = torch.bmm(k.unsqueeze(1), memory_matrix)

            v = self.projector(torch.cat((v, v_h.squeeze(1)), dim=-1))
            v = self.ln2(v)
        
        k = k.unsqueeze(2)
        v = v.unsqueeze(1)
        
        hebb = self.gamma_pos * (self.w_max - memory_matrix) * k * v - self.gamma_neg * memory_matrix * k**2

        memory_matrix = hebb + memory_matrix

        return memory_matrix
    
class ReadingCell(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(ReadingCell, self).__init__()
        self.projector_k = nn.Linear(in_features=config.embedding_size+config.memory_size, 
                                     out_features=config.memory_size, 
                                     bias=config.use_bias)
        nn.init.kaiming_uniform_(self.projector_k.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, question: torch.Tensor, memory_matrix: torch.Tensor, states: torch.Tensor):
        v = states

        k = self.projector_k(torch.cat((question, v), dim=-1))

        v = torch.bmm(k.unsqueeze(1), memory_matrix)
        
        return v.squeeze(1)
