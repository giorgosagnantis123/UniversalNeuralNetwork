# -*- coding: utf-8 -*-
"""UniversalNeuralNetwork.ipynb

"""

import numpy as np
import random

"""one network version"""
class QuantumWeightSuperposition:
   def __init__(self, initial_weight, max_candidates):
     """Initializes the quantum weight pool, tracking potential candidate weights,
     their imaginary parts (uncertainty/volatility), and their utiltiy scores"""

     self.candidates=[initial_weight]
     self.imaginary_parts=[np.random.uniform(0.01, 0.05)]
     self.max_candidates= max_candidates


     self.last_selected_index=0
     self.last_selected_weight= initial_weight
     self.candidate_utility=[1.0]


   def select_weight_entangled(self, current_probability_residue, annealing_temp=1.0):
     """Selects a weight from the superposition based on a probabiity distribution derived
     from the weight magnitudes, uncertainties, and utilities, modulated by temperature"""

     w_sq= np.array(self.candidates)**2
     r_sq= np.array(self.imaginary_parts)**2


     magnitudes= (w_sq/ (np.mean(w_sq) + 1e-6))+ (r_sq/ (np.mean(r_sq)+ 1e-6)) +(0.1 * np.array(self.candidate_utility))

     temp=max(0.01, current_probability_residue * annealing_temp)

     shift_energies=magnitudes-np.max(magnitudes)
     exp_energies=np.exp(shift_energies/temp)
     probabilities=exp_energies/np.sum(exp_energies)

     idx=np.random.choice(len(self.candidates),p=probabilities )

     self.last_selected_index=idx
     self.last_selected_weight=self.candidates[idx]
     return self.last_selected_weight,probabilities[idx]

   def update_weight_superposition(self, ideal_weight):
     """Updates the pool of candidate weights by either adding a newly discovered ideal weight
     (if limits allow) or adjusting the imaginary parts and utilities of existing candidates"""

     candidates_arr=np.array(self.candidates)
     differences= np.abs(candidates_arr - ideal_weight)
     best_candidate_idx= np.argmin(differences)


     is_new_discovery=differences[best_candidate_idx]>1e-4
     if is_new_discovery:
        if len(self.candidates)<self.max_candidates:
            self.candidates.append(ideal_weight)
            self.imaginary_parts.append(0.05)
            self.candidate_utility.append(1.0)
            best_candidate_idx=len(self.candidates)-1

        else:
            weakest_idx= np.argmin(self.imaginary_parts)
            self.candidates[weakest_idx]=ideal_weight
            self.imaginary_parts[weakest_idx]=0.05
            self.candidate_utility[weakest_idx]=1.0
            best_candidate_idx= weakest_idx




     chosen_idx = self.last_selected_index
     r_learning_rate=0.06


     self.imaginary_parts[best_candidate_idx]+= r_learning_rate

     self.candidate_utility[best_candidate_idx]=min(10, self.candidate_utility[best_candidate_idx]+0.5)


     if best_candidate_idx != chosen_idx:

        self.imaginary_parts[chosen_idx] = max(0, self.imaginary_parts[chosen_idx] - r_learning_rate)
        self.candidate_utility[chosen_idx]*= 0.95
     else:

        for i in range(len(self.imaginary_parts)):
           if i!= chosen_idx:
             self.imaginary_parts[i] = max(0, self.imaginary_parts[i] - (r_learning_rate * 0.1))

   def refresh_candidate_pool(self, merge_threshold=0.05,mutation_scale=0.03):
         """Merges similar weight candidates to reduce redundancy and allows for the creation of new weigths to explore new values
          and prevent pool stagnation"""

         W=np.array(self.candidates)
         R=np.array(self.imaginary_parts)
         U=np.array(self.candidate_utility)

         n=len(W)
         if n<=1:
            return
         order=np.argsort(W)

         W=W[order]
         R=R[order]
         U=U[order]

         merged_W=[]
         merged_R=[]
         merged_U=[]

         i=0
         while i<n:

               if i+1<n:

                   if abs(W[i] - W[i+1]) <= merge_threshold and np.sign(W[i])== np.sign(W[i+1]):

                       merged_W.append(0.5 * (W[i]+W[i+1]))
                       merged_R.append(max(R[i] , R[i+1]))
                       merged_U.append(max(U[i], U[i+1]))

                       i+=2
                       continue

               merged_W.append(W[i])
               merged_R.append(R[i])
               merged_U.append(U[i])
               i += 1


         self.candidates=merged_W
         self.imaginary_parts=merged_R
         self.candidate_utility= merged_U

         current_size= len(self.candidates)
         open_positions=self.max_candidates- current_size

         if open_positions>0:

            W_arr= np.array(self.candidates)
            R_arr= np.array(self.imaginary_parts)
            U_arr= np.array(self.candidate_utility)

            masses=(W_arr **2) + (R_arr **2) +(0.1* U_arr)
            best_idx=np.argmax(masses)

            w_best=self.candidates[best_idx]

            for k in range(open_positions):

                if k % 2==0:
                   w_new= w_best + np.random.normal(0.0, mutation_scale)
                   r_new=0.05
                   u_new=1.0

                else:
                   w_new=np.random.uniform(-0.5, 0.5)
                   r_new=0.05
                   u_new=1.0

                self.candidates.append(w_new)
                self.imaginary_parts.append(r_new)
                self.candidate_utility.append(u_new)



class NovelNetwork:

  def __init__(self, input_dimension):
    """Initializes the neural network architecture, hyper-parameters,
    and tracking variables for the moving average loss"""

    self.input_dimension=input_dimension
    self.layer_neuron_counts=[input_dimension]
    self.weights= []
    self.entanglement_ratio=0.05
    self.max_weight_limit=21.0
    self.moving_avg_loss=5.0
    self.loss_alpha=0.1
    self._initialize_network_structure()

  def _initialize_network_structure(self):
    """Builds the neural network structure, dynamically sizing layers and
    instantiating QuantumWeightSuperposition objects for each synaptic connection"""

    current_size= self.input_dimension


    while current_size< (2*self.input_dimension)**2 and len(self.layer_neuron_counts)<5:
      next_layer_size=2*current_size
      self.layer_neuron_counts.append(next_layer_size)
      current_size=next_layer_size

    for i in range(len(self.layer_neuron_counts)-1):
      num_neurons=self.layer_neuron_counts[i]
      layer_connections= []
      limit=np.sqrt(6/(num_neurons*2))


      for j in range(num_neurons):
          neuron_connections= []
          n_connections=2


          for _ in range(n_connections):
             w_init=np.random.uniform(-limit,limit)
             qc= QuantumWeightSuperposition(initial_weight=w_init, max_candidates=(1 * self.input_dimension+1))
             neuron_connections.append(qc)

          layer_connections.append(neuron_connections)

      self.weights.append(layer_connections)

  def forward(self, input_vector):
      """Performs a forward pass through the network, selecting entangled weights,
      aggregating activations, and applying max pooling alongside a Leaky Relu-Like activation"""

      activations={'layer_activations':[np.array(input_vector).flatten()], 'pooled_values':[], 'pooled_indices':[]}
      current_layer=activations['layer_activations'][0]

      annealing_temp=np.tanh(self.moving_avg_loss/2.0)
      annealing_temp=max(0.01, annealing_temp)

      for i, layer_conn_pool in enumerate(self.weights):
          next_layer=[]
          all_connections_flat=[]

          for neuron_connections in layer_conn_pool:
              for qc in neuron_connections:
                  all_connections_flat.append(qc)

          probability_sum_per_weight=max(0.1, 1.0-self.entanglement_ratio)

          for qc in all_connections_flat:

             _, _ = qc.select_weight_entangled(probability_sum_per_weight,annealing_temp)

          for j in range(len(current_layer)):
            neuron_value=current_layer[j]
            neuron_connections=layer_conn_pool[j]

            for qc in neuron_connections:

                selected_w=qc.last_selected_weight
                next_layer.append(neuron_value * selected_w)


          next_layer=np.array(next_layer)
          current_layer=next_layer
          activations['layer_activations'].append(current_layer)


      final_output=current_layer

      pooled_values=[]
      pooled_indices=[]

      n=len(final_output)
      pool_group_index=0


      i=0
      while i<n:
             end_idx=min(i+2, n)
             group=final_output[i:end_idx]

             if len(group)>0:

               if pool_group_index % 2==0:
                pool_val=np.max(group)
                local_idx= np.argmax(group)
               else:
                 pool_val=np.max(group)
                 local_idx= np.argmax(group)

               pooled_values.append(pool_val)
               pooled_indices.append(i+local_idx)

             else:
               pooled_values.append(0.0)
               pooled_indices.append(i)

             pool_group_index +=1
             i+=2

      activations['pooled_values']=np.array(pooled_values)
      activations['pooled_indices']=pooled_indices

      small_constant=1e-6

      alpha=0.07
      pools=activations["pooled_values"]+small_constant
      relu_values=np.where(pools>0,pools, alpha*pools)

      prediction= np.sum(relu_values)

      return prediction, activations, relu_values

  def multi_layer_propagation(self, x_input, y_true):
      """Excecutes the backward pass (learning phase), calculating target distributions
      for each layer and updating the quantum weight superpositions to minimize error"""

      y_pred, activations, _ =self.forward(x_input)

      y_pred=y_pred.item()
      target=y_true.item()
      small_constant=1e-6


      current_error=abs(y_pred-target)
      self.moving_avg_loss=(self.loss_alpha *current_error)+(1-self.loss_alpha)* self.moving_avg_loss


      pooled_vals=activations['pooled_values']
      indices=activations['pooled_indices']
      scaling_factor=target/(y_pred+small_constant) if np.sum(pooled_vals)>0 else 1.0
      desired_pooled=pooled_vals * scaling_factor if np.sum(pooled_vals)>0 else np.full_like(pooled_vals, target/len(pooled_vals))

      final_layer=activations['layer_activations'][-1]

      desired_output=np.zeros_like(final_layer)

      for pool_idx, real_idx in enumerate(indices):
          desired_output[real_idx] = desired_pooled[pool_idx]


      for layer_idx in range(len(self.weights)-1,-1,-1):
          layer_output=activations['layer_activations'][layer_idx]
          current_layer_connections=self.weights[layer_idx]

          updated_output=np.zeros_like(layer_output)
          flat_conn_index=0

          for neuron_idx in range(len(current_layer_connections)):
              neuron_value=layer_output[neuron_idx]
              connections=current_layer_connections[neuron_idx]
              output_count=len(connections)

              beginning= flat_conn_index
              end=beginning+ output_count
              targets_for_destination=desired_output[beginning:end]
              effective_targets= targets_for_destination

              new_weights_for_projection=[]

              for k, qc in enumerate(connections):

                  target_val=effective_targets[k]
                  w_used=qc.last_selected_weight

                  idx_used=qc.last_selected_index
                  uncertainty=qc.imaginary_parts[idx_used]
                  utility=qc.candidate_utility[idx_used]
                  alpha_i=np.clip(uncertainty/(utility+1e-2), 0.1, 0.9)

                  denom= neuron_value if abs(neuron_value)>1e-4 else 1e-4 * np.sign(neuron_value+1e-9)
                  w_ideal=target_val/denom


                  h=0.04

                  w_new= w_used +(alpha_i*h*np.tanh(current_error)) *(w_ideal-w_used)

                  w_final=self.max_weight_limit* np.tanh(w_new/self.max_weight_limit)

                  qc.update_weight_superposition(w_final)
                  new_weights_for_projection.append(w_final)

              W_arr=np.array(new_weights_for_projection)
              T_arr=np.array(effective_targets)

              numerator=np.sum(T_arr*W_arr)
              denominator=np.sum(W_arr**2)+1e-6

              n_new=numerator/denominator

              updated_output[neuron_idx]=n_new
              flat_conn_index +=output_count

          desired_output=updated_output

  def refresh_all_quantum_pools(self):
       """Triggers the candidate pool refresh mechanism across all quantum connections
       in the network using an adaptive mutation scale based on recent loss"""

       adaptive_mutation=0.03 +(0.05* (1.0- np.tanh(self.moving_avg_loss)))
       for layer in self.weights:
           for neuron_connections in layer:
               for qc in neuron_connections:
                   qc.refresh_candidate_pool(mutation_scale=adaptive_mutation)

  def get_network_diagnostics(self):
      """Calculates and returns network health metrics, including average weight diversity,
      volatility, and superposition depth(entropy)"""

      total_entropy=0
      total_volatility=0
      active_candidates=0
      synapse_count=0
      e=1e-9

      for layer in self.weights:
          for neuron_pool in layer:
              for qc in neuron_pool:
                  w=np.array(qc.candidates)
                  std_dev= np.std(w)
                  total_entropy+= std_dev

                  r=np.array(qc.imaginary_parts)
                  total_volatility+= np.mean(r)

                  u= np.array(qc.candidate_utility)
                  m= (w/ (np.mean(w) + 1e-6))+ (r/ (np.mean(r)+ 1e-6)) +(0.1 * u)
                  probs=np.exp(m-np.max(m))/np.sum(np.exp(m-np.max(m)))

                  pool_h=-np.sum(probs*np.log(probs+e))
                  active_candidates+=pool_h
                  synapse_count+=1

      avg_diversity=total_entropy/synapse_count
      avg_volatility=total_volatility/synapse_count
      avg_active=active_candidates/synapse_count
      return f" Average Weight Diversity: {avg_diversity:.4f}, Volatility:{avg_volatility:.4f}, superposition depth: {avg_active:.2f}"


  def train(self, x_train, y_train, epochs):
      """Runs the training loop over a specified number of epochs, performing forward
      and backward passes,and periodically refreshing quantum pools"""

      for epoch in range(epochs):

        total_loss_a=0

        health_a=self.get_network_diagnostics()


        print(f"Network A pool diversity: {health_a}")

        print("")
        print("- - - - - - - - - - - - - - -")



        for x, y_true in zip(x_train, y_train):

             pred_a, _, features_from_a=self.forward(x)




             loss_a=abs(pred_a-y_true)

             total_loss_a += loss_a.item()




             self.multi_layer_propagation(x, y_true)



        avg_loss_a=total_loss_a/ len(x_train)


        print(f"Epoch{epoch:2d}, average loss network a {avg_loss_a:.4f}")
        if epoch%5==0:
           self.refresh_all_quantum_pools()

      return  avg_loss_a


if __name__ == "__main__":
  def create_data(num_samples=2100, num_features=3):
   """Generates a syntetic, non-linear dataset for training and testing"""

   x_data=np.random.rand(num_samples, num_features)*2
   y_data=(x_data[:, 0]**2+np.sin(20 *x_data[:, 1] *x_data[:, 0])+x_data[:, 2]**3)
   y_data=y_data.reshape(-1,1)

   return x_data, y_data

  INPUT_FEATURES=3
  Epochs=15
  np.random.seed(42)


  x_set, y_set = create_data(num_samples=2100, num_features=INPUT_FEATURES)
  x_train, y_train=x_set[:1500], y_set[:1500]
  x_test, y_test=x_set[1500:], y_set[1500:]


  np.random.seed(42)
  print("Starting Training")
  print(f"Training on {len(x_train)} samples for {Epochs} epochs")
  double_network=NovelNetwork(INPUT_FEATURES)
  final_avr_loss=double_network.train(x_train, y_train, epochs=Epochs)

  print(" ")
  print("Training complete")
  print("")

  test_errors_a=[]
  test_errors=[]
  for x_val, y_val in zip(x_test, y_test):
      pred_a, _, features_from_a=double_network.forward(x_val)


      error_a=abs(pred_a-y_val.item())


      test_errors_a.append(error_a)

  print(f"Average test loss net_a: {np.mean(test_errors_a):.4f}")
  print(f"Min test error net_a: {np.min(test_errors_a):.4f}")
  print(f"Max test error net_a: {np.max(test_errors_a):.4f}")
