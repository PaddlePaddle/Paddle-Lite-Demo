// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "graphbuilderadapter.h"
#include "yaml-cpp/contrib/graphbuilder.h"

namespace YAML {
struct Mark;

int GraphBuilderAdapter::ContainerFrame::sequenceMarker;

void GraphBuilderAdapter::OnNull(const Mark &mark, anchor_t anchor) {
  void *pParent = GetCurrentParent();
  void *pNode = m_builder.NewNull(mark, pParent);
  RegisterAnchor(anchor, pNode);

  DispositionNode(pNode);
}

void GraphBuilderAdapter::OnAlias(const Mark &mark, anchor_t anchor) {
  void *pReffedNode = m_anchors.Get(anchor);
  DispositionNode(m_builder.AnchorReference(mark, pReffedNode));
}

void GraphBuilderAdapter::OnScalar(const Mark &mark, const std::string &tag,
                                   anchor_t anchor, const std::string &value) {
  void *pParent = GetCurrentParent();
  void *pNode = m_builder.NewScalar(mark, tag, pParent, value);
  RegisterAnchor(anchor, pNode);

  DispositionNode(pNode);
}

void GraphBuilderAdapter::OnSequenceStart(const Mark &mark,
                                          const std::string &tag,
                                          anchor_t anchor,
                                          EmitterStyle::value /* style */) {
  void *pNode = m_builder.NewSequence(mark, tag, GetCurrentParent());
  m_containers.push(ContainerFrame(pNode));
  RegisterAnchor(anchor, pNode);
}

void GraphBuilderAdapter::OnSequenceEnd() {
  void *pSequence = m_containers.top().pContainer;
  m_containers.pop();

  DispositionNode(pSequence);
}

void GraphBuilderAdapter::OnMapStart(const Mark &mark, const std::string &tag,
                                     anchor_t anchor,
                                     EmitterStyle::value /* style */) {
  void *pNode = m_builder.NewMap(mark, tag, GetCurrentParent());
  m_containers.push(ContainerFrame(pNode, m_pKeyNode));
  m_pKeyNode = nullptr;
  RegisterAnchor(anchor, pNode);
}

void GraphBuilderAdapter::OnMapEnd() {
  void *pMap = m_containers.top().pContainer;
  m_pKeyNode = m_containers.top().pPrevKeyNode;
  m_containers.pop();
  DispositionNode(pMap);
}

void *GraphBuilderAdapter::GetCurrentParent() const {
  if (m_containers.empty()) {
    return nullptr;
  }
  return m_containers.top().pContainer;
}

void GraphBuilderAdapter::RegisterAnchor(anchor_t anchor, void *pNode) {
  if (anchor) {
    m_anchors.Register(anchor, pNode);
  }
}

void GraphBuilderAdapter::DispositionNode(void *pNode) {
  if (m_containers.empty()) {
    m_pRootNode = pNode;
    return;
  }

  void *pContainer = m_containers.top().pContainer;
  if (m_containers.top().isMap()) {
    if (m_pKeyNode) {
      m_builder.AssignInMap(pContainer, m_pKeyNode, pNode);
      m_pKeyNode = nullptr;
    } else {
      m_pKeyNode = pNode;
    }
  } else {
    m_builder.AppendToSequence(pContainer, pNode);
  }
}
}  // namespace YAML
