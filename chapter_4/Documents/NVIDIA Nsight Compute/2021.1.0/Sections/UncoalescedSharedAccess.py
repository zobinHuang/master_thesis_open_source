# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import NvRules
import sys

def get_identifier():
    return "UncoalescedSharedAccess"

def get_name():
    return "Uncoalesced Shared Accesses"

def get_description():
    return "Uncoalesced Shared Accesses"

def get_section_identifier():
    return "SourceCounters"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    shared_sectors_metric = action.metric_by_name("memory_l1_sectors_shared")
    ideal_shared_sectors_metric = action.metric_by_name("memory_l1_sectors_shared_ideal")
    total_shared_sectors = shared_sectors_metric.as_uint64()
    total_ideal_shared_sectors = ideal_shared_sectors_metric.as_uint64()
    # No need to check further if total shared sectors match with the ideal value
    if total_shared_sectors <= total_ideal_shared_sectors:
        return

    num_shared_sectors_instances = shared_sectors_metric.num_instances()
    num_ideal_shared_sectors_instances = ideal_shared_sectors_metric.num_instances()
    # We cannot execute the rule if we don't get the same instance count for both metrics
    if num_shared_sectors_instances != num_ideal_shared_sectors_instances:
        return

    shared_corr_ids = shared_sectors_metric.correlation_ids()

    shared_diff = {}
    for i in range(num_shared_sectors_instances):
        per_instance_shared_sectors = shared_sectors_metric.as_uint64(i)
        per_instance_ideal_shared_sectors = ideal_shared_sectors_metric.as_uint64(i)
        if (per_instance_shared_sectors != per_instance_ideal_shared_sectors):
            shared_diff[i] = abs(per_instance_ideal_shared_sectors - per_instance_shared_sectors)

    # Sort the data w.r.t difference between shared sectors and ideal sectors and show top 10 entries
    shared_sorted_diff = sorted(shared_diff.items(), key=lambda kv: kv[1], reverse=True)
    num_entries = min(10, len(shared_sorted_diff))

    for i in range(num_entries):
        instance = shared_sorted_diff[i][0]
        per_instance_shared_sectors = shared_sectors_metric.as_uint64(instance)
        per_instance_ideal_shared_sectors = ideal_shared_sectors_metric.as_uint64(instance)
        corr_id = shared_corr_ids.as_uint64(instance)

        message = "Uncoalesced shared access, expected {:d} sectors, got {:d} ({:.2f}x) at PC @source:{:x}:0x{:x}@"\
        .format(per_instance_ideal_shared_sectors, per_instance_shared_sectors, per_instance_shared_sectors / per_instance_ideal_shared_sectors, corr_id, corr_id)
        source_info = action.source_info(corr_id)
        if source_info != None:
            message += " at {:s}:{:d}".format(source_info.file_name(), source_info.line())

        fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)

